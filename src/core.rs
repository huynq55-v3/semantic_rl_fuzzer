use rand::seq::IndexedRandom;
use rand::RngExt;
use rayon::prelude::*;
use std::collections::HashSet;
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
    pub masks: Vec<Vec<Vec<bool>>>,
    pub log_probs: Vec<f32>,
    pub reward: f32,
    pub is_interesting: bool,
    pub is_crash: bool,
}

// 🌟 BẢN FIX VÔ THƯỢNG: Lưu trữ toàn bộ "linh hồn" của ván game
#[derive(Clone, Debug)]
pub struct SavedSeed<E: FuzzEnvironment> {
    pub env: E,
    pub states_history: Vec<E::State>,
    pub actions_history: Vec<E::Action>,
    pub action_indices_history: Vec<Vec<usize>>,
    pub masks_history: Vec<Vec<Vec<bool>>>,
    pub log_probs_history: Vec<f32>, // Cần thiết để hàm learn_from_batch không bị crash
    pub accumulated_reward: f32,
}

pub struct FuzzCorpus<E: FuzzEnvironment> {
    pub interesting_seeds: Vec<Trajectory<E::State, E::Action>>,
    pub seen_states: HashSet<u64>,
    pub saved_envs: Vec<SavedSeed<E>>,
}

impl<E: FuzzEnvironment> FuzzCorpus<E> {
    pub fn new() -> Self {
        Self {
            interesting_seeds: Vec::new(),
            seen_states: HashSet::new(),
            saved_envs: Vec::new(),
        }
    }
}

pub trait FuzzEnvironment: Clone + Send + Sync {
    type State: Send + Sync + Clone;
    type Action: Send + Sync;

    fn get_state(&self) -> Self::State;
    fn get_action_mask(&self) -> Vec<Vec<bool>>;
    fn step(&mut self, action: &Self::Action) -> StepResult<Self::State>;
    fn reset(&mut self);
    fn hash_state(state: &Self::State) -> u64;
}

pub trait TruthOracle<E: FuzzEnvironment>: Send + Sync {
    fn judge(&self, env: &mut E, is_invalid: bool) -> OracleStatus;
}

pub trait FuzzActor: Send + Clone {
    type State;
    type Action;

    fn choose_action(
        &self,
        state_history: &[Self::State],
        masks: &[Vec<bool>],
    ) -> (Self::Action, Vec<usize>, f32);

    fn choose_batch_action(
        &self,
        state_histories: &[Vec<Self::State>],
        masks_batch: &[Vec<Vec<bool>>],
    ) -> Vec<(Self::Action, Vec<usize>, f32)>;
}

pub trait NeuralAgent: Send {
    type State;
    type Action;
    type Actor: FuzzActor<State = Self::State, Action = Self::Action>;

    fn get_actor(&self) -> Self::Actor;
    fn learn_from_batch(&mut self, trajectories: &[Trajectory<Self::State, Self::Action>]);
}

pub struct FuzzEngine<
    E: FuzzEnvironment,
    A: NeuralAgent<State = E::State, Action = E::Action>,
    O: TruthOracle<E>,
> {
    pub base_env: E,
    pub agent: A,
    pub oracle: O,
    pub corpus: FuzzCorpus<E>,
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
        let mut rng = rand::rng();

        for iteration in 1..=self.config.total_iterations {
            let start_time = Instant::now();

            let actor = self.agent.get_actor();
            let oracle_ref = &self.oracle;
            let num_envs = self.config.num_envs;
            let max_steps = self.config.max_steps_per_episode;

            let mut envs: Vec<E> = vec![self.base_env.clone(); num_envs];

            let mut rollouts: Vec<Trajectory<E::State, E::Action>> = vec![
                Trajectory {
                    states: Vec::with_capacity(max_steps + 1),
                    actions: Vec::with_capacity(max_steps),
                    action_indices: Vec::with_capacity(max_steps),
                    masks: Vec::with_capacity(max_steps),
                    log_probs: Vec::with_capacity(max_steps),
                    reward: 0.0,
                    is_interesting: false,
                    is_crash: false,
                };
                num_envs
            ];

            // 🌟 KHÔI PHỤC KÝ ỨC (Amnesia Fix)
            for i in 0..num_envs {
                let env = &mut envs[i];
                let traj = &mut rollouts[i];

                if !self.corpus.saved_envs.is_empty() && rng.random_bool(0.5) {
                    let seed = self.corpus.saved_envs.choose(&mut rng).unwrap();
                    *env = seed.env.clone();

                    // Khôi phục đồng bộ mọi mảng dữ liệu
                    traj.states = seed.states_history.clone();
                    traj.actions = seed.actions_history.clone();
                    traj.action_indices = seed.action_indices_history.clone();
                    traj.masks = seed.masks_history.clone();
                    traj.log_probs = seed.log_probs_history.clone();
                    traj.reward = seed.accumulated_reward;
                } else {
                    env.reset();
                }
            }

            let mut active_mask = vec![true; num_envs];

            for _step in 0..max_steps {
                let current_states: Vec<E::State> =
                    envs.par_iter().map(|e| e.get_state()).collect();

                let current_histories: Vec<Vec<E::State>> = rollouts
                    .iter()
                    .zip(current_states.iter())
                    .map(|(t, s)| {
                        let mut hist = t.states.clone();
                        hist.push(s.clone());
                        hist
                    })
                    .collect();

                let current_masks: Vec<Vec<Vec<bool>>> =
                    envs.par_iter().map(|e| e.get_action_mask()).collect();

                let batch_results = actor.choose_batch_action(&current_histories, &current_masks);

                let step_results: Vec<_> = envs
                    .par_iter_mut()
                    .zip(current_states.into_par_iter())
                    .zip(current_masks.into_par_iter())
                    .zip(batch_results.into_par_iter())
                    .zip(active_mask.par_iter())
                    .map(
                        |(
                            (((env, state_before), mask_before), (action, indices, log_prob)),
                            &is_active,
                        )| {
                            if !is_active {
                                return None;
                            }
                            let result = env.step(&action);
                            let status = oracle_ref.judge(env, result.is_invalid);
                            Some((
                                state_before,
                                action,
                                indices,
                                mask_before,
                                log_prob,
                                result.next_state,
                                status,
                                env.clone(),
                            ))
                        },
                    )
                    .collect();

                let mut any_active = false;
                for (i, res) in step_results.into_iter().enumerate() {
                    if let Some((s_before, act, idx, mask, lp, s_next, status, env_snapshot)) = res
                    {
                        let traj = &mut rollouts[i];

                        // Chỉ push s_before nếu traj.states đang rỗng (mới reset)
                        // Nếu load từ Seed, traj.states đã có dữ liệu, ta không push s_before nữa
                        if traj.states.is_empty() {
                            traj.states.push(s_before);
                        }

                        traj.actions.push(act);
                        traj.action_indices.push(idx);
                        traj.masks.push(mask);
                        traj.log_probs.push(lp);
                        traj.states.push(s_next.clone()); // Push state mới vào cuối mảng

                        let hash_val = E::hash_state(&s_next);
                        if self.corpus.seen_states.insert(hash_val) {
                            traj.reward += 1.0;
                            traj.is_interesting = true;

                            // 🌟 LƯU TOÀN BỘ LINH HỒN
                            self.corpus.saved_envs.push(SavedSeed {
                                env: env_snapshot,
                                states_history: traj.states.clone(),
                                actions_history: traj.actions.clone(),
                                action_indices_history: traj.action_indices.clone(),
                                masks_history: traj.masks.clone(),
                                log_probs_history: traj.log_probs.clone(),
                                accumulated_reward: traj.reward,
                            });
                        }

                        match status {
                            OracleStatus::Violated => {
                                traj.is_interesting = true;
                                traj.is_crash = true;
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
                // Tổng số bước đi sẽ tính dựa trên chiều dài của mảng actions
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
                        let content = format!(
                            "Action sequence (Length: {}):\n{:#?}",
                            traj.actions.len(),
                            traj.actions
                        );
                        let _ = artifact_tx.send((filename, content));
                    }
                }
            }

            // Gửi dữ liệu ĐỒNG BỘ TUYỆT ĐỐI cho NeuralAgent học
            self.agent.learn_from_batch(&rollouts);

            if iteration % self.config.log_interval == 0 {
                let avg_reward = total_batch_reward / num_envs as f32;
                let elapsed = start_time.elapsed().as_secs_f64();
                let fps = total_steps_taken as f64 / elapsed;

                println!("📊 [Iter {} | Ep {}] Avg Reward: {:.2} | Crashes: {} | State Coverage: {} (Saved: {}) | Speed: {:.0} steps/s",
                    iteration, total_episodes, avg_reward, crashes_found, self.corpus.seen_states.len(), self.corpus.saved_envs.len(), fps
                );
                on_log(iteration, &rollouts);
            }
        }
        drop(artifact_tx);
        let _ = writer_thread.join();
    }
}
