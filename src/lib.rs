use rayon::prelude::*;
use std::sync::mpsc;
use std::time::Instant;

// ==========================================
// PART 1: CORE STRUCTURES & CONFIG
// ==========================================

#[derive(Debug, Clone)]
pub struct FuzzConfig {
    pub num_envs: usize, // Parallel environments (e.g., 1024)
    pub max_steps_per_episode: usize,
    pub total_iterations: usize, // Total training cycles
    pub log_interval: usize,     // Log every N iterations
}

#[derive(Debug, Clone)]
pub enum OracleStatus {
    Hold { reward: f32 },
    Violated,
    Invalid,
}

/// The result returned by the Environment after executing a step.
/// Contains only execution facts — semantic judgment is delegated to the TruthOracle.
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

/// On-policy buffer: Clear after training iteration
pub struct RolloutBuffer<S, A> {
    pub trajectories: Vec<Trajectory<S, A>>,
}

impl<S, A> RolloutBuffer<S, A> {
    pub fn new() -> Self {
        Self {
            trajectories: Vec::new(),
        }
    }
    pub fn clear(&mut self) {
        self.trajectories.clear();
    }
}

/// Persistent storage for high-value seeds (crashes or high coverage)
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

// ==========================================
// PART 2: THE INTERFACES (CONTRACTS)
// The boundaries between the AI, the Target, the Oracle, and the Fuzzer Core.
// ==========================================

/// The interface that any fuzzing target (e.g., Bevy, Sled) must implement.
/// Step should only execute the action — semantic judgment is delegated to TruthOracle.
// Yêu cầu Clone, Send, Sync để Rayon có thể nhân bản Env ra nhiều luồng
pub trait FuzzEnvironment: Clone + Send + Sync {
    type State: Send + Sync;
    type Action: Send + Sync;

    fn get_state(&self) -> Self::State;
    fn get_action_mask(&self) -> Vec<bool>;
    fn step(&mut self, action: &Self::Action) -> StepResult<Self::State>;
    fn reset(&mut self);
}

pub trait TruthOracle<E: FuzzEnvironment>: Send + Sync {
    fn judge(&self, env: &mut E, is_invalid: bool) -> OracleStatus;
}

/// Lightweight, thread-safe actor for inference across Rayon threads.
/// Stripped of Autodiff overhead — only pure forward computation.
pub trait FuzzActor: Send + Clone {
    type State;
    type Action;
    fn choose_action(&self, state: &Self::State, mask: &[bool]) -> (Self::Action, Vec<usize>, f32);
}

/// The learner side: owns the Autodiff graph, produces lightweight Actors.
/// Only requires `Send` (not `Sync`) since it is never shared across threads.
pub trait NeuralAgent: Send {
    type State;
    type Action;
    type Actor: FuzzActor<State = Self::State, Action = Self::Action>;

    /// Extract a thread-safe actor (no Autodiff) for parallel inference.
    fn get_actor(&self) -> Self::Actor;
    fn learn_from_batch(&mut self, trajectories: &[Trajectory<Self::State, Self::Action>]);
}

// ==========================================
// PART 3: THE ORCHESTRATOR
// Binds the Environment, the Agent, and the Oracle together.
// ==========================================

pub struct FuzzEngine<
    E: FuzzEnvironment,
    A: NeuralAgent<State = E::State, Action = E::Action>,
    O: TruthOracle<E>,
> {
    pub base_env: E, // Template env to clone for each thread
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

        let mut global_episodes = 0;

        for iteration in 1..=self.config.total_iterations {
            let start_time = Instant::now();

            // ==========================================
            // EXTRACT ACTOR (thread-safe, no Autodiff)
            // Pre-clone on main thread to avoid Sync requirement
            // ==========================================
            let actor_template = self.agent.get_actor();
            let oracle_ref = &self.oracle;
            let max_steps = self.config.max_steps_per_episode;

            // Pre-clone actors + envs on the main thread (single-threaded, no Sync needed)
            let work_items: Vec<_> = (0..self.config.num_envs)
                .map(|_| (actor_template.clone(), self.base_env.clone()))
                .collect();

            let mut rollouts: Vec<Trajectory<E::State, E::Action>> = work_items
                .into_par_iter()
                .map(|(actor, mut local_env)| {
                    local_env.reset();

                    let mut traj = Trajectory {
                        states: Vec::with_capacity(max_steps),
                        actions: Vec::with_capacity(max_steps),
                        action_indices: Vec::with_capacity(max_steps),
                        log_probs: Vec::with_capacity(max_steps),
                        reward: 0.0,
                        is_interesting: false,
                    };

                    for _ in 0..max_steps {
                        let state = local_env.get_state();
                        let mask = local_env.get_action_mask();

                        // Use ACTOR (owned, not shared) for thread-safe inference
                        let (action, indices, log_prob) = actor.choose_action(&state, &mask);
                        let result = local_env.step(&action);

                        traj.states.push(state.clone());
                        traj.actions.push(action.clone());
                        traj.action_indices.push(indices);
                        traj.log_probs.push(log_prob);

                        let status = oracle_ref.judge(&mut local_env, result.is_invalid);

                        match status {
                            OracleStatus::Violated => {
                                traj.is_interesting = true;
                                break;
                            }
                            OracleStatus::Hold { reward } => traj.reward += reward,
                            OracleStatus::Invalid => {
                                traj.reward -= 1.0;
                                break;
                            }
                        }
                    }

                    if traj.reward > 0.0 {
                        traj.is_interesting = true;
                    }
                    traj
                })
                .collect();

            global_episodes += self.config.num_envs;

            // ==========================================
            // XỬ LÝ KẾT QUẢ & TRAIN
            // ==========================================
            let mut total_batch_reward = 0.0;
            let mut crashes_found = 0;

            for traj in rollouts.iter() {
                total_batch_reward += traj.reward;

                if traj.is_interesting {
                    if traj.reward == 0.0 && traj.states.is_empty() { // Example crash detection
                         // Logic defined by your environment
                    }
                    // For now, if Violated was reached or reward > 0
                    if traj.reward.is_nan() {
                        continue;
                    } // Safety

                    // If it's a crash (Violated), we likely want to save it
                    // The Trajectory doesn't explicitly store the OracleStatus it ended with,
                    // but we can infer or add it. Let's keep it simple.
                    self.corpus.interesting_seeds.push(traj.clone());
                }
            }

            // Re-detect crashes for artifact saving
            for (i, traj) in rollouts.iter().enumerate() {
                if traj.is_interesting && traj.reward <= -1.0 {
                    // Simplified crash/invalid indicator
                    crashes_found += 1;
                    let filename = format!("artifacts/bug_iter_{}_env_{}.txt", iteration, i);
                    let content = format!("Action sequence:\n{:#?}", traj.actions);
                    let _ = artifact_tx.send((filename, content));
                }
            }

            // ==========================================
            // LOGGING & CALLBACK
            // ==========================================
            if iteration % self.config.log_interval == 0 {
                let avg_reward = total_batch_reward / self.config.num_envs as f32;
                let elapsed = start_time.elapsed().as_secs_f64();
                let fps =
                    (self.config.num_envs * self.config.max_steps_per_episode) as f64 / elapsed;

                println!(
                    "📊 [Iter {} | Ep {}] Avg Reward: {:.2} | Crashes: {} | Speed: {:.0} steps/s",
                    iteration, global_episodes, avg_reward, crashes_found, fps
                );

                // Fire callback so consuming code can analyze the batch
                on_log(iteration, &rollouts);
            }

            self.agent.learn_from_batch(&rollouts);
            rollouts.clear();
        }

        drop(artifact_tx);
        let _ = writer_thread.join();
    }
}

#[cfg(feature = "burn-backend")]
pub mod burn_helpers {
    // 1. Đổi sang NdArray cho CPU optimization
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;
    use burn::module::AutodiffModule;
    use burn::nn::{Linear, LinearConfig, Relu};
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};
    use burn::prelude::*;
    use burn::tensor::backend::AutodiffBackend;
    use rand::distr::{weighted::WeightedIndex, Distribution};
    use rand::rng;
    use rayon::prelude::*;

    // Sử dụng NdArray làm backend mặc định
    pub type DefaultCpuBackend = Autodiff<NdArray>;

    #[derive(Module, Debug)]
    pub struct MultiHeadNet<B: Backend> {
        shared_layer_1: Linear<B>,
        shared_layer_2: Linear<B>,
        heads: Vec<Linear<B>>,
        relu: Relu,
    }

    impl<B: Backend> MultiHeadNet<B> {
        pub fn new(
            device: &B::Device,
            input_size: usize,
            hidden_size: usize,
            head_sizes: &[usize],
        ) -> Self {
            let shared_layer_1 = LinearConfig::new(input_size, hidden_size).init(device);
            let shared_layer_2 = LinearConfig::new(hidden_size, hidden_size).init(device);

            let mut heads = Vec::new();
            for &size in head_sizes {
                heads.push(LinearConfig::new(hidden_size, size).init(device));
            }

            Self {
                shared_layer_1,
                shared_layer_2,
                heads,
                relu: Relu::new(),
            }
        }

        pub fn forward(&self, state: Tensor<B, 2>) -> Vec<Tensor<B, 2>> {
            let x = self.shared_layer_1.forward(state);
            let x = self.relu.forward(x);
            let x = self.shared_layer_2.forward(x);
            let shared_features = self.relu.forward(x);

            self.heads
                .iter()
                .map(|head| head.forward(shared_features.clone()))
                .collect()
        }
    }

    pub trait ActionTranslator: Send + Sync + Clone {
        type TargetAction: Send + Sync + Clone + std::fmt::Debug;
        fn translate(&self, head_outputs: &[usize]) -> Self::TargetAction;
    }

    pub struct BurnAgent<B: AutodiffBackend, T: ActionTranslator, O: Optimizer<MultiHeadNet<B>, B>> {
        pub net: MultiHeadNet<B>,
        pub translator: T,
        pub optimizer: O,
        pub learning_rate: f64,
        pub device: B::Device,
    }

    // ==========================================
    // BURN ACTOR: Thread-safe inference (no Autodiff)
    // ==========================================
    #[derive(Clone)]
    pub struct BurnActor<B: Backend, T: ActionTranslator> {
        pub net: MultiHeadNet<B>,
        pub translator: T,
        pub device: B::Device,
    }

    impl<B: Backend, T: ActionTranslator> super::FuzzActor for BurnActor<B, T> {
        type State = Vec<f32>;
        type Action = T::TargetAction;

        fn choose_action(
            &self,
            state: &Self::State,
            mask: &[bool],
        ) -> (Self::Action, Vec<usize>, f32) {
            let state_tensor = Tensor::<B, 2>::from_data(
                TensorData::new(state.clone(), [1, state.len()]),
                &self.device,
            );

            let head_logits = self.net.forward(state_tensor);

            let mut selected_indices = Vec::new();
            let mut total_log_prob = 0.0;

            for (i, mut logits) in head_logits.into_iter().enumerate() {
                if i == 0 && !mask.is_empty() {
                    let mask_data = TensorData::new(mask.to_vec(), [1, mask.len()]);
                    let mask_tensor = Tensor::<B, 2, Bool>::from_data(mask_data, &self.device);
                    logits = logits.mask_fill(mask_tensor.bool_not(), -1e9);
                }

                let probs = burn::tensor::activation::softmax(logits, 1);
                let probs_vec = probs
                    .into_data()
                    .to_vec::<f32>()
                    .expect("Failed to get Tensor data");

                let mut rng = rng();
                let dist = WeightedIndex::new(&probs_vec).unwrap();
                let chosen_index = dist.sample(&mut rng);

                selected_indices.push(chosen_index);
                total_log_prob += probs_vec[chosen_index].ln();
            }

            let final_action = self.translator.translate(&selected_indices);
            (final_action, selected_indices, total_log_prob)
        }
    }

    // ==========================================
    // BURN AGENT: Learner side (owns Autodiff graph)
    // ==========================================
    impl<B: AutodiffBackend, T: ActionTranslator, O: Optimizer<MultiHeadNet<B>, B>>
        super::NeuralAgent for BurnAgent<B, T, O>
    {
        type State = Vec<f32>;
        type Action = T::TargetAction;
        type Actor = BurnActor<B::InnerBackend, T>;

        fn get_actor(&self) -> Self::Actor {
            BurnActor {
                net: self.net.valid(), // .valid() strips Autodiff → plain Backend
                translator: self.translator.clone(),
                device: self.device.clone(),
            }
        }

        fn learn_from_batch(
            &mut self,
            trajectories: &[super::Trajectory<Self::State, Self::Action>],
        ) {
            // 1. CHUẨN BỊ DỮ LIỆU CỰC NHANH (Flattened + Parallel)
            let (all_states, all_action_indices, all_rewards): (Vec<f32>, Vec<Vec<i64>>, Vec<f32>) =
                trajectories
                    .par_iter()
                    .filter(|t| !t.states.is_empty() && t.reward != 0.0)
                    .map(|t| {
                        let mut s = Vec::new();
                        let mut ai = Vec::new();
                        let mut r = Vec::new();
                        for (i, state) in t.states.iter().enumerate() {
                            s.extend_from_slice(state);
                            ai.push(
                                t.action_indices[i]
                                    .iter()
                                    .map(|&idx| idx as i64)
                                    .collect::<Vec<_>>(),
                            );
                            r.push(t.reward);
                        }
                        (s, ai, r)
                    })
                    .reduce(
                        || (Vec::new(), Vec::new(), Vec::new()),
                        |mut acc, (s, ai, r)| {
                            acc.0.extend(s);
                            acc.1.extend(ai);
                            acc.2.extend(r);
                            acc
                        },
                    );

            if all_states.is_empty() {
                return;
            }

            let total_steps = all_rewards.len();
            let state_dim = trajectories[0].states[0].len();

            // 2. CHUYỂN THÀNH TENSOR KHỔNG LỒ (NdArray + MKL optimization)
            let states_tensor = Tensor::<B, 2>::from_data(
                TensorData::new(all_states, [total_steps, state_dim]),
                &self.device,
            );
            let rewards_tensor = Tensor::<B, 1>::from_data(
                TensorData::new(all_rewards, [total_steps]),
                &self.device,
            );

            // 3. MỘT LẦN FORWARD DUY NHẤT CHO TOÀN BỘ BATCH
            let all_head_logits = self.net.forward(states_tensor);

            let mut total_loss = Tensor::<B, 1>::from_data([0.0], &self.device);

            // 4. TÍNH LOSS VECTORIZED
            for (h, logits) in all_head_logits.into_iter().enumerate() {
                let probs = burn::tensor::activation::softmax(logits, 1);

                let current_head_indices: Vec<i64> = all_action_indices
                    .iter()
                    .map(|indices| indices[h])
                    .collect();
                let index_tensor = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(current_head_indices, [total_steps, 1]),
                    &self.device,
                );

                let log_probs = probs.gather(1, index_tensor).log().reshape([total_steps]);
                let head_loss = log_probs.mul(rewards_tensor.clone()).neg().mean();
                total_loss = total_loss.add(head_loss);
            }

            // 5. MỘT LẦN BACKWARD DUY NHẤT
            let gradients = total_loss.backward();
            let grads = GradientsParams::from_grads(gradients, &self.net);

            // 6. UPDATE WEIGHTS
            self.net = self
                .optimizer
                .step(self.learning_rate, self.net.clone(), grads);

            println!(
                "🔥 Trained batch of {} steps. CPU running at full capacity.",
                total_steps
            );
        }
    }

    pub fn create_cpu_agent<T: ActionTranslator>(
        input_size: usize,
        hidden_size: usize,
        head_sizes: &[usize],
        learning_rate: f64,
        translator: T,
    ) -> BurnAgent<
        DefaultCpuBackend,
        T,
        impl Optimizer<MultiHeadNet<DefaultCpuBackend>, DefaultCpuBackend>,
    > {
        let device = Default::default();

        let net =
            MultiHeadNet::<DefaultCpuBackend>::new(&device, input_size, hidden_size, head_sizes);

        let optimizer = AdamConfig::new().init();

        BurnAgent {
            net,
            translator,
            optimizer,
            learning_rate,
            device,
        }
    }
}
