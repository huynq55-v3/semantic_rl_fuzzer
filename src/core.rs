use rayon::prelude::*;
use std::sync::mpsc;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct FuzzConfig {
    pub num_envs: usize,
    pub max_steps_per_episode: usize,
    pub total_iterations: usize,
    pub log_interval: usize,
}

#[derive(Debug, Clone)]
pub enum OracleStatus {
    Hold { reward: f32 },
    Violated,
    Invalid,
}

pub struct StepResult<S> {
    pub next_state: S,
    pub is_invalid: bool,
}

#[derive(Clone, Debug)]
pub struct Trajectory<S, A> {
    pub states: Vec<S>,
    pub actions: Vec<A>,
    pub action_indices: Vec<Vec<usize>>,
    pub log_probs: Vec<f32>,
    pub reward: f32,
    pub is_interesting: bool,
}

pub struct FuzzCorpus<S, A> {
    pub interesting_seeds: Vec<Trajectory<S, A>>,
}

impl<S, A> FuzzCorpus<S, A> {
    pub fn new() -> Self {
        Self {
            interesting_seeds: Vec::new(),
        }
    }
}

pub trait FuzzEnvironment: Clone + Send + Sync {
    type State: Send + Sync;
    type Action: Send + Sync;

    fn get_state(&self) -> Self::State;
    fn get_action_mask(&self) -> Vec<Vec<bool>>;
    fn step(&mut self, action: &Self::Action) -> StepResult<Self::State>;
    fn reset(&mut self);
}

pub trait TruthOracle<E: FuzzEnvironment>: Send + Sync {
    fn judge(&self, env: &mut E, is_invalid: bool) -> OracleStatus;
}

pub trait FuzzActor: Send + Clone {
    type State;
    type Action;

    fn choose_action(
        &self,
        state: &Self::State,
        masks: &[Vec<bool>],
    ) -> (Self::Action, Vec<usize>, f32);

    fn choose_batch_action(
        &self,
        states: &[Self::State],
        masks_batch: &[Vec<Vec<bool>>],
    ) -> Vec<(Self::Action, Vec<usize>, f32)>;
}

pub trait NeuralAgent: Send {
    type State;
    type Action;
    type Actor: FuzzActor<State = Self::State, Action = Self::Action>;

    fn get_actor(&self) -> Self::Actor;
    fn learn_from_batch(&mut self, trajectories: &[Trajectory<Self::State, Self::Action>]) -> f32;
    fn reset_forward_net(&mut self);
    fn get_curiosity_threshold(&self) -> f32;
}

pub struct FuzzEngine<
    E: FuzzEnvironment,
    A: NeuralAgent<State = E::State, Action = E::Action>,
    O: TruthOracle<E>,
> {
    pub base_env: E,
    pub agent: A,
    pub oracle: O,
    pub corpus: FuzzCorpus<E::State, E::Action>,
    pub config: FuzzConfig,
}

impl<E, A, O> FuzzEngine<E, A, O>
where
    E: FuzzEnvironment + 'static,
    E::State: Clone,
    E::Action: Clone + std::fmt::Debug,
    A: NeuralAgent<State = E::State, Action = E::Action>,
    O: TruthOracle<E>,
{
    pub fn run_fuzzing<L>(&mut self, mut on_log: L)
    where
        L: FnMut(usize, &[Trajectory<E::State, E::Action>]),
    {
        let (artifact_tx, artifact_rx) = mpsc::channel::<(String, String)>();

        let writer_thread = std::thread::spawn(move || {
            let _ = std::fs::create_dir_all("artifacts");
            for (filename, content) in artifact_rx {
                let _ = std::fs::write(&filename, content);
            }
        });

        let mut total_episodes = 0;

        for iteration in 1..=self.config.total_iterations {
            let start_time = Instant::now();

            let actor = self.agent.get_actor();
            let oracle_ref = &self.oracle;
            let num_envs = self.config.num_envs;
            let max_steps = self.config.max_steps_per_episode;

            let mut envs: Vec<E> = vec![self.base_env.clone(); num_envs];
            for env in envs.iter_mut() {
                env.reset();
            }

            let mut rollouts: Vec<Trajectory<E::State, E::Action>> = vec![
                Trajectory {
                    states: Vec::with_capacity(max_steps * 2),
                    actions: Vec::with_capacity(max_steps),
                    action_indices: Vec::with_capacity(max_steps),
                    log_probs: Vec::with_capacity(max_steps),
                    reward: 0.0,
                    is_interesting: false,
                };
                num_envs
            ];

            let mut active_mask = vec![true; num_envs];

            for _step in 0..max_steps {
                let current_states: Vec<E::State> =
                    envs.par_iter().map(|e| e.get_state()).collect();
                let current_masks: Vec<Vec<Vec<bool>>> =
                    envs.par_iter().map(|e| e.get_action_mask()).collect();

                let batch_results = actor.choose_batch_action(&current_states, &current_masks);

                let step_results: Vec<_> = envs
                    .par_iter_mut()
                    .zip(current_states.into_par_iter())
                    .zip(batch_results.into_par_iter())
                    .zip(active_mask.par_iter())
                    .map(
                        |(((env, state_before), (action, indices, log_prob)), &is_active)| {
                            if !is_active {
                                return None;
                            }

                            let result = env.step(&action);
                            let status = oracle_ref.judge(env, result.is_invalid);

                            Some((
                                state_before,
                                action,
                                indices,
                                log_prob,
                                result.next_state,
                                status,
                            ))
                        },
                    )
                    .collect();

                let mut any_active = false;
                for (i, res) in step_results.into_iter().enumerate() {
                    if let Some((s_before, act, idx, lp, s_next, status)) = res {
                        let traj = &mut rollouts[i];

                        traj.states.push(s_before);
                        traj.actions.push(act);
                        traj.action_indices.push(idx);
                        traj.log_probs.push(lp);
                        traj.states.push(s_next);

                        match status {
                            OracleStatus::Violated => {
                                traj.is_interesting = true;
                                active_mask[i] = false;
                            }
                            OracleStatus::Hold { reward } => {
                                traj.reward += reward;
                                any_active = true;
                            }
                            OracleStatus::Invalid => {
                                traj.reward -= 1.0;
                                active_mask[i] = false;
                            }
                        }
                    }
                }

                if !any_active {
                    break;
                }
            }

            total_episodes += num_envs;

            let mut total_batch_reward = 0.0;
            let mut crashes_found = 0;
            let mut total_steps_taken = 0;

            for (i, traj) in rollouts.iter_mut().enumerate() {
                total_steps_taken += traj.actions.len();
                total_batch_reward += traj.reward;

                if traj.reward > 0.0 {
                    traj.is_interesting = true;
                }

                if traj.is_interesting {
                    if !traj.reward.is_nan() {
                        self.corpus.interesting_seeds.push(traj.clone());
                    }

                    if traj.reward <= -1.0 || traj.states.is_empty() {
                        crashes_found += 1;
                        let filename = format!("artifacts/bug_iter_{}_env_{}.txt", iteration, i);
                        let content = format!("Action sequence:\n{:#?}", traj.actions);
                        let _ = artifact_tx.send((filename, content));
                    }
                }
            }

            let current_curiosity = self.agent.learn_from_batch(&rollouts);

            if iteration % self.config.log_interval == 0 {
                let avg_reward = total_batch_reward / num_envs as f32;
                let elapsed = start_time.elapsed().as_secs_f64();
                let fps = total_steps_taken as f64 / elapsed;

                println!(
                    "[Iter {} | Ep {}] Avg Reward: {:.2} | Crashes: {} | Speed: {:.0} steps/s",
                    iteration, total_episodes, avg_reward, crashes_found, fps
                );

                if current_curiosity < self.agent.get_curiosity_threshold() && crashes_found == 0 {
                    self.agent.reset_forward_net();
                }

                on_log(iteration, &rollouts);
            }
        }

        drop(artifact_tx);
        let _ = writer_thread.join();
    }
}
