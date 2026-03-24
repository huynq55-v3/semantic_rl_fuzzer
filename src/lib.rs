use rand::prelude::IndexedRandom;
use rand::rng;

// ==========================================
// PART 1: CORE STRUCTURES (IMMUTABLE)
// These structs handle the flow and data storage,
// independent of any specific fuzzing target.
// ==========================================

/// Represents the verdict from the Truth Oracle after evaluating an action.
#[derive(Debug, Clone)]
pub enum OracleStatus {
    Hold { reward: f32 }, // Normal execution, returns an intrinsic/extrinsic reward
    Violated,             // LOGIC BUG FOUND! The system violated an invariant.
    Invalid,              // The action was invalid (e.g., syntax error, out of bounds).
}

/// The result returned by the Environment after executing a step.
/// Contains only execution facts — semantic judgment is delegated to the TruthOracle.
pub struct StepResult<S> {
    pub next_state: S,
    pub is_invalid: bool,
}

/// A sequence of states and actions representing a single episode.
#[derive(Clone, Debug)]
pub struct Trajectory<S, A> {
    pub states: Vec<S>,
    pub actions: Vec<A>,
    pub action_indices: Vec<Vec<usize>>, // For Neural agents to re-forward correctly
    pub log_probs: Vec<f32>,             // Neural network's confidence for calculating loss
    pub reward: f32,                     // Total accumulated reward
    pub is_interesting: bool,            // Flag to keep this trajectory as a future mutation seed
}

/// A priority-based storage for past trajectories.
pub struct HybridReplayBuffer<S, A> {
    pub capacity: usize,
    pub memory: Vec<Trajectory<S, A>>,
}

impl<S: Clone, A: Clone> HybridReplayBuffer<S, A> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            memory: Vec::with_capacity(capacity),
        }
    }

    /// Pushes a new trajectory into the buffer. Evicts using O(1) swap_remove.
    pub fn push_trajectory(&mut self, traj: Trajectory<S, A>) {
        if self.memory.len() >= self.capacity {
            // Find the FIRST "trash" element (not interesting) to evict
            if let Some(idx) = self.memory.iter().position(|t| !t.is_interesting) {
                self.memory.swap_remove(idx); // O(1) instead of O(N)
            } else {
                // If the entire buffer consists of high-quality trajectories (rare)
                self.memory.swap_remove(0); // O(1) instead of O(N)
            }
        }
        self.memory.push(traj);
    }

    /// Randomly samples a batch of trajectories for Neural Network backpropagation.
    pub fn sample_for_training(&self, batch_size: usize) -> Vec<Trajectory<S, A>> {
        let mut rng = rng();
        self.memory.sample(&mut rng, batch_size).cloned().collect()
    }
}

// ==========================================
// PART 2: THE INTERFACES (CONTRACTS)
// The boundaries between the AI, the Target, the Oracle, and the Fuzzer Core.
// ==========================================

/// The interface that any fuzzing target (e.g., Bevy, Sled) must implement.
/// Step should only execute the action — semantic judgment is delegated to TruthOracle.
pub trait FuzzEnvironment {
    type State;
    type Action;

    /// Returns the current physical/semantic state of the target.
    fn get_state(&self) -> Self::State;

    /// Returns a boolean mask indicating which actions are currently valid.
    fn get_action_mask(&self) -> Vec<bool>;

    /// Executes the action and returns the execution result.
    /// Should NOT evaluate semantic correctness — that's the Oracle's job.
    fn step(&mut self, action: &Self::Action) -> StepResult<Self::State>;

    /// Resets the environment for a new episode.
    fn reset(&mut self);
}

/// The supreme judge. Completely decoupled from the Environment.
/// Evaluates whether the Environment is violating any physical/logical invariants.
pub trait TruthOracle<E: FuzzEnvironment> {
    /// Judges the current state of the environment after an action was executed.
    /// `is_invalid` indicates whether the action was syntactically invalid.
    fn judge(&self, env: &mut E, is_invalid: bool) -> OracleStatus;
}

/// The interface for the AI model (e.g., a Burn Tensor Neural Network).
pub trait NeuralAgent {
    type State;
    type Action;

    /// Takes the current state and valid action mask, returns the chosen action,
    /// indices (for multi-head), and its log probability.
    fn choose_action(&self, state: &Self::State, mask: &[bool]) -> (Self::Action, Vec<usize>, f32);

    /// Triggers the backpropagation process using a batch of past experiences.
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
    pub env: E,
    pub agent: A,
    pub oracle: O,
    pub buffer: HybridReplayBuffer<E::State, E::Action>,
    pub max_steps_per_episode: usize,
    pub batch_size: usize,
}

impl<E: FuzzEnvironment, A: NeuralAgent<State = E::State, Action = E::Action>, O: TruthOracle<E>>
    FuzzEngine<E, A, O>
where
    E::State: Clone,
    E::Action: Clone + std::fmt::Debug,
{
    /// Starts the main Reinforcement Learning fuzzing loop.
    pub fn run_fuzzing(&mut self, total_episodes: usize) {
        // Background thread for non-blocking artifact writing
        let (artifact_tx, artifact_rx) = std::sync::mpsc::channel::<(String, String)>();
        let writer_thread = std::thread::spawn(move || {
            use std::fs;
            use std::io::Write;
            let _ = fs::create_dir_all("artifacts");
            for (filename, content) in artifact_rx {
                if let Ok(mut file) = fs::File::create(&filename) {
                    let _ = write!(file, "{}", content);
                }
            }
        });

        for episode in 1..=total_episodes {
            self.env.reset();

            let mut current_traj = Trajectory {
                states: Vec::new(),
                actions: Vec::new(),
                action_indices: Vec::new(),
                log_probs: Vec::new(),
                reward: 0.0,
                is_interesting: false,
            };

            for _step in 0..self.max_steps_per_episode {
                let state = self.env.get_state();
                let mask = self.env.get_action_mask();

                let (action, indices, log_prob) = self.agent.choose_action(&state, &mask);
                let result = self.env.step(&action);

                current_traj.states.push(state.clone());
                current_traj.actions.push(action.clone());
                current_traj.action_indices.push(indices);
                current_traj.log_probs.push(log_prob);

                // DELEGATE JUDGMENT TO THE ORACLE
                let status = self.oracle.judge(&mut self.env, result.is_invalid);

                match status {
                    OracleStatus::Violated => {
                        println!(
                            "🚨 [Episode {}] LOGIC BUG DETECTED! Terminating this episode!",
                            episode
                        );
                        current_traj.is_interesting = true;

                        // Send artifact to background writer (non-blocking)
                        let filename = format!("artifacts/bug_ep_{}.txt", episode);
                        let mut content =
                            String::from("Action sequence that crashed the system:\n");
                        for (i, a) in current_traj.actions.iter().enumerate() {
                            content.push_str(&format!("{}. {:?}\n", i, a));
                        }
                        let _ = artifact_tx.send((filename.clone(), content));
                        println!("💾 Evidence queued for saving to {}", filename);

                        break;
                    }
                    OracleStatus::Hold { reward } => {
                        current_traj.reward += reward;
                    }
                    OracleStatus::Invalid => {
                        current_traj.reward -= 1.0;
                        break;
                    }
                }
            }

            // Mark trajectories with positive returns as interesting seeds.
            if current_traj.reward > 0.0 {
                current_traj.is_interesting = true;
            }

            // PRINT DETAILED LOGS (Every 50 Episodes)
            if episode % 50 == 0 {
                println!("\n🔍 [DEBUG Episode {}] AI Mind Revealed:", episode);
                println!("  - Total Reward: {:.2}", current_traj.reward);
                println!(
                    "  - Sequence Length (Steps): {}",
                    current_traj.actions.len()
                );
                println!("  - Last 3 actions chosen by AI:");

                let tail_len = current_traj.actions.len().min(3);
                let start_idx = current_traj.actions.len() - tail_len;

                for (i, action) in current_traj.actions[start_idx..].iter().enumerate() {
                    let prob_percent = current_traj.log_probs[start_idx + i].exp() * 100.0;
                    println!(
                        "      + Command: {:?} (Confidence: {:.2}%)",
                        action, prob_percent
                    );
                }
            }

            self.buffer.push_trajectory(current_traj);

            if episode % self.batch_size == 0 {
                let batch = self.buffer.sample_for_training(self.batch_size);
                self.agent.learn_from_batch(&batch);

                println!(
                    "🧠 [Episode {}] Finished learning a batch of {} samples from the replay buffer.",
                    episode, self.batch_size
                );
            }
        }

        // Signal the background writer to finish and wait for it
        drop(artifact_tx);
        let _ = writer_thread.join();
    }
}

#[cfg(feature = "burn-backend")]
pub mod burn_helpers {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::backend::Autodiff;
    use burn::nn::{Linear, LinearConfig, Relu};
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};
    use burn::prelude::*;
    use burn::tensor::backend::AutodiffBackend;
    use rand::distr::{weighted::WeightedIndex, Distribution};
    use rand::rng;

    /// Default GPU Backend for Intel Iris Xe or other WGPU-compatible GPUs.
    pub type DefaultGpuBackend = Autodiff<Wgpu>;

    #[derive(Module, Debug)]
    pub struct MultiHeadNet<B: Backend> {
        shared_layer_1: Linear<B>,
        shared_layer_2: Linear<B>,
        heads: Vec<Linear<B>>,
        relu: Relu,
    }

    impl<B: Backend> MultiHeadNet<B> {
        /// Initializes the Neural Network with a dynamic number of heads.
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

        /// Returns a list of Logit Tensors for each head.
        /// Input shape: [batch, features] → Output: Vec of [batch, head_size] per head.
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

    /// ACTION TRANSLATOR TRAIT
    pub trait ActionTranslator {
        type TargetAction;
        fn translate(&self, head_outputs: &[usize]) -> Self::TargetAction;
    }

    /// AGENT WRAPPER (Automates all AI-related logic)
    pub struct BurnAgent<B: AutodiffBackend, T: ActionTranslator, O: Optimizer<MultiHeadNet<B>, B>>
    {
        pub net: MultiHeadNet<B>,
        pub translator: T,
        pub optimizer: O,
        pub learning_rate: f64,
        pub device: B::Device,
    }

    impl<B: AutodiffBackend, T: ActionTranslator, O: Optimizer<MultiHeadNet<B>, B>>
        super::NeuralAgent for BurnAgent<B, T, O>
    {
        type State = Vec<f32>;
        type Action = T::TargetAction;

        fn choose_action(
            &self,
            state: &Self::State,
            mask: &[bool],
        ) -> (Self::Action, Vec<usize>, f32) {
            let state_tensor =
                Tensor::<B, 1>::from_data(state.as_slice(), &self.device).unsqueeze::<2>();

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
                let probs_vec = probs.into_data().to_vec::<f32>().unwrap();

                let mut rng = rng();
                let dist = WeightedIndex::new(&probs_vec).unwrap();
                let chosen_index = dist.sample(&mut rng);

                selected_indices.push(chosen_index);
                total_log_prob += probs_vec[chosen_index].ln();
            }

            let final_action = self.translator.translate(&selected_indices);
            (final_action, selected_indices, total_log_prob)
        }

        fn learn_from_batch(
            &mut self,
            trajectories: &[super::Trajectory<Self::State, Self::Action>],
        ) {
            let batch_size = trajectories.len();
            if batch_size == 0 {
                return;
            }

            // STEP 1: Reconstruct Tensor Graph from memory (VECTORIZED)
            let mut loss_tensors = Vec::new();

            for traj in trajectories {
                if traj.reward < 0.0 {
                    continue;
                }

                let step_count = traj.states.len();
                if step_count == 0 {
                    continue;
                }

                let state_feature_size = traj.states[0].len();

                // VECTORIZED: Flatten all states into [step_count, features] — ONE GPU upload
                let all_states_flat: Vec<f32> =
                    traj.states.iter().flat_map(|s| s.iter().copied()).collect();
                let states_tensor =
                    Tensor::<B, 1>::from_data(all_states_flat.as_slice(), &self.device)
                        .reshape([step_count, state_feature_size]);

                // ONE batched forward pass — GPU processes all steps at once
                let all_head_logits = self.net.forward(states_tensor);

                let episode_return = traj.reward;
                let reward_tensor =
                    Tensor::<B, 1>::from_data([episode_return], &self.device);

                // For each head, compute losses in batch using vectorized gather
                for (h, logits) in all_head_logits.into_iter().enumerate() {
                    // logits shape: [step_count, head_size]
                    let probs = burn::tensor::activation::softmax(logits, 1);

                    // Build index tensor for this head across all steps: [step_count, 1]
                    let indices: Vec<i64> =
                        traj.action_indices.iter().map(|ai| ai[h] as i64).collect();
                    let index_tensor =
                        Tensor::<B, 1, Int>::from_data(indices.as_slice(), &self.device)
                            .reshape([step_count, 1]);

                    // Gather all chosen probabilities at once: [step_count, 1]
                    let chosen_probs = probs.gather(1, index_tensor);
                    let log_probs = chosen_probs.log();

                    // REINFORCE: Loss = -(Reward * LogProb)
                    // Broadcast reward across all steps in this trajectory
                    let step_losses =
                        log_probs.mul(reward_tensor.clone().unsqueeze::<2>()).neg();
                    loss_tensors.push(step_losses.reshape([step_count]));
                }
            }

            if loss_tensors.is_empty() {
                return;
            }

            // STEP 2: Aggregate losses and calculate the mean
            let total_loss = Tensor::cat(loss_tensors, 0).mean();

            let loss_value = total_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            println!(
                "🧠 [Training] Backpropagating... Current Loss: {:.4}",
                loss_value
            );

            // STEP 3: Trigger Backpropagation
            let gradients = total_loss.backward();

            // STEP 4: Update weights using the Optimizer
            let grads = GradientsParams::from_grads(gradients, &self.net);
            self.net = self
                .optimizer
                .step(self.learning_rate, self.net.clone(), grads);

            println!("✅ Model weights updated! The AI is a bit smarter now.");
        }
    }

    /// GPU AGENT FACTORY (FACADE)
    pub fn create_gpu_agent<T: ActionTranslator>(
        input_size: usize,
        hidden_size: usize,
        head_sizes: &[usize],
        learning_rate: f64,
        translator: T,
    ) -> BurnAgent<
        DefaultGpuBackend,
        T,
        impl Optimizer<MultiHeadNet<DefaultGpuBackend>, DefaultGpuBackend>,
    > {
        let device = WgpuDevice::DefaultDevice;

        let net =
            MultiHeadNet::<DefaultGpuBackend>::new(&device, input_size, hidden_size, head_sizes);

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
