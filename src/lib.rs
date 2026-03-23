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
pub struct StepResult<S> {
    pub next_state: S,
    pub reward: f32,
    pub is_violated: bool,
    pub is_invalid: bool,
}

/// A sequence of states and actions representing a single episode.
#[derive(Clone, Debug)]
pub struct Trajectory<S, A> {
    pub states: Vec<S>,
    pub actions: Vec<A>,
    pub action_indices: Vec<Vec<usize>>, // For Neural agents to re-forward correctly
    pub log_probs: Vec<f32>,              // Neural network's confidence for calculating loss
    pub reward: f32,                      // Total accumulated reward
    pub is_interesting: bool,             // Flag to keep this trajectory as a future mutation seed
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

    /// Pushes a new trajectory into the buffer. Removes the oldest if at capacity.
    pub fn push_trajectory(&mut self, traj: Trajectory<S, A>) {
        if self.memory.len() >= self.capacity {
            // Simplified FIFO eviction.
            // In a production environment, sort and evict the lowest reward first.
            self.memory.remove(0);
        }
        self.memory.push(traj);
    }

    /// Checks if the buffer has enough data to form a training batch.
    pub fn is_ready_for_batch(&self, batch_size: usize) -> bool {
        self.memory.len() >= batch_size
    }

    /// Randomly samples a batch of trajectories for Neural Network backpropagation.
    pub fn sample_for_training(&self, batch_size: usize) -> Vec<Trajectory<S, A>> {
        let mut rng = rng();
        self.memory.sample(&mut rng, batch_size).cloned().collect()
    }
}

// ==========================================
// PART 2: THE INTERFACES (CONTRACTS)
// The boundaries between the AI, the Target, and the Fuzzer Core.
// ==========================================

/// The interface that any fuzzing target (e.g., Bevy, Sled) must implement.
pub trait FuzzEnvironment {
    type State;
    type Action;

    /// Returns the current physical/semantic state of the target.
    fn get_state(&self) -> Self::State;

    /// Returns a boolean mask indicating which actions are currently valid.
    fn get_action_mask(&self) -> Vec<bool>;

    /// Executes the action, evaluates it against the Oracle, and returns the result.
    fn step(&mut self, action: &Self::Action) -> StepResult<Self::State>;

    /// Resets the environment for a new episode.
    fn reset(&mut self);
}

/// The interface for the AI model (e.g., a Burn Tensor Neural Network).
pub trait NeuralAgent {
    type State;
    type Action;

    /// Takes the current state and valid action mask, returns the chosen action, indices (for multi-head), and its log probability.
    fn choose_action(&self, state: &Self::State, mask: &[bool]) -> (Self::Action, Vec<usize>, f32);

    /// Triggers the backpropagation process using a batch of past experiences.
    fn learn_from_batch(&mut self, trajectories: &[Trajectory<Self::State, Self::Action>]);
}

// ==========================================
// PART 3: THE ORCHESTRATOR
// Binds the Environment and the Agent together.
// ==========================================

pub struct FuzzEngine<E: FuzzEnvironment, A: NeuralAgent<State = E::State, Action = E::Action>> {
    pub env: E,
    pub agent: A,
    pub buffer: HybridReplayBuffer<E::State, E::Action>,
    pub max_steps_per_episode: usize,
    pub batch_size: usize,
}

impl<E: FuzzEnvironment, A: NeuralAgent<State = E::State, Action = E::Action>> FuzzEngine<E, A>
where
    E::State: Clone,
    E::Action: Clone,
{
    /// Starts the main Reinforcement Learning fuzzing loop.
    pub fn run_fuzzing(&mut self, total_episodes: usize) {
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
                current_traj.reward += result.reward;

                if result.is_violated {
                    println!(
                        "🚨 [Episode {}] LOGIC BUG DETECTED! Terminating episode to save artifact.",
                        episode
                    );
                    current_traj.is_interesting = true;
                    break;
                }

                if result.is_invalid {
                    // Penalize the AI for generating an invalid action and abort the sequence.
                    current_traj.reward -= 1.0;
                    break;
                }
            }

            // Mark trajectories with positive returns as interesting seeds for future mutations.
            if current_traj.reward > 0.0 {
                current_traj.is_interesting = true;
            }
            self.buffer.push_trajectory(current_traj);

            // Trigger Neural Network training if we have gathered enough episodes.
            if self.buffer.is_ready_for_batch(self.batch_size) {
                let batch = self.buffer.sample_for_training(self.batch_size);
                self.agent.learn_from_batch(&batch);
                println!(
                    "🧠 [Episode {}] Batch training complete. Backpropagation applied!",
                    episode
                );
            }
        }
    }
}

#[cfg(feature = "burn-backend")]
pub mod burn_helpers {
    use burn::nn::{Linear, LinearConfig, Relu};
    use burn::optim::{GradientsParams, Optimizer};
    use burn::prelude::*;
    use burn::tensor::backend::AutodiffBackend;
    use rand::distr::{weighted::WeightedIndex, Distribution};
    use rand::rng;

    #[derive(Module, Debug)]
    pub struct MultiHeadNet<B: Backend> {
        shared_layer_1: Linear<B>,
        shared_layer_2: Linear<B>,
        // List of output layers (Heads). Each head is responsible for generating one parameter.
        heads: Vec<Linear<B>>,
        relu: Relu,
    }

    impl<B: Backend> MultiHeadNet<B> {
        /// Initializes the Neural Network with a dynamic number of heads.
        /// Example: head_sizes = &[4, 100, 50] -> Will create 3 heads.
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
        pub fn forward(&self, state: Tensor<B, 2>) -> Vec<Tensor<B, 2>> {
            let x = self.shared_layer_1.forward(state);
            let x = self.relu.forward(x);
            let x = self.shared_layer_2.forward(x);
            let shared_features = self.relu.forward(x);

            // Each head receives the same set of features but generates different outputs.
            self.heads
                .iter()
                .map(|head| head.forward(shared_features.clone()))
                .collect()
        }
    }

    /// 2. ACTION TRANSLATOR (TRANSLATOR TRAIT)
    /// Users must implement this trait to convert the AI's integer array into their own action Enum.
    pub trait ActionTranslator {
        type TargetAction;

        /// Takes an array of IDs from the heads (e.g., [0, 42, 1]) -> Returns the actual Enum.
        fn translate(&self, head_outputs: &[usize]) -> Self::TargetAction;
    }

    /// 3. AGENT WRAPPER (Automates all AI-related logic)
    pub struct BurnAgent<B: AutodiffBackend, T: ActionTranslator, O: Optimizer<MultiHeadNet<B>, B>> {
        pub net: MultiHeadNet<B>,
        pub translator: T,
        pub optimizer: O,
        pub learning_rate: f64,
        pub device: B::Device,
    }

    // Automatically implement the standard NeuralAgent from the core library.
    impl<B: AutodiffBackend, T: ActionTranslator, O: Optimizer<MultiHeadNet<B>, B>> super::NeuralAgent
        for BurnAgent<B, T, O>
    {
        type State = Vec<f32>; // Input must be a float array
        type Action = T::TargetAction;

        fn choose_action(
            &self,
            state: &Self::State,
            mask: &[bool],
        ) -> (Self::Action, Vec<usize>, f32) {
            // 1. Convert Vec<f32> to Tensor [Batch=1, Features]
            let state_tensor =
                Tensor::<B, 1>::from_data(state.as_slice(), &self.device).unsqueeze::<2>();

            // 2. Run the Neural Network (Forward pass)
            let head_logits = self.net.forward(state_tensor);

            let mut selected_indices = Vec::new();
            let mut total_log_prob = 0.0;

            // 3. Process each Head
            for (i, mut logits) in head_logits.into_iter().enumerate() {
                // ACTION MASKING TECHNIQUE
                if i == 0 && !mask.is_empty() {
                    let mask_data = TensorData::new(mask.to_vec(), [1, mask.len()]);
                    let mask_tensor = Tensor::<B, 2, Bool>::from_data(mask_data, &self.device);
                    logits = logits.mask_fill(mask_tensor.bool_not(), -1e9);
                }

                // Convert Logits to Probabilities (Softmax)
                let probs = burn::tensor::activation::softmax(logits, 1);
                let probs_vec = probs.into_data().to_vec::<f32>().unwrap();

                // Sampling randomly based on probabilities
                let mut rng = rng();
                let dist = WeightedIndex::new(&probs_vec).unwrap();
                let chosen_index = dist.sample(&mut rng);

                selected_indices.push(chosen_index);

                // Cumulative LogProb for REINFORCE calculation
                total_log_prob += probs_vec[chosen_index].ln();
            }

            // 4. Translate the array [0, 42, 1] into the system Enum using the translator
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

            // STEP 1: Reconstruct Tensor Graph from memory
            let mut loss_tensors = Vec::new();

            for traj in trajectories {
                // Skip invalid sequences
                if traj.reward < 0.0 {
                    continue;
                }

                let step_count = traj.states.len();
                if step_count == 0 {
                    continue;
                }

                // Use the total episode return as the weight for REINFORCE
                let episode_return = traj.reward;
                let reward_tensor = Tensor::<B, 1>::from_data([episode_return], &self.device);

                // Re-forward pass for each step in the episode
                for i in 0..step_count {
                    let state = &traj.states[i];
                    let action_indices = &traj.action_indices[i];

                    let state_tensor = Tensor::<B, 1>::from_data(state.as_slice(), &self.device)
                        .unsqueeze::<2>();

                    // Forward pass with Autodiff enabled
                    let head_logits = self.net.forward(state_tensor);

                    // For each head, calculate the log probability of the action that was actually taken
                    for (h, logits) in head_logits.into_iter().enumerate() {
                        let probs = burn::tensor::activation::softmax(logits, 1);
                        
                        // Select the probability of the index that was chosen during the episode
                        let chosen_index = action_indices[h];
                        let chosen_index_tensor = Tensor::<B, 1, Int>::from_data(
                            [chosen_index as i64], 
                            &self.device
                        ).unsqueeze::<2>();
                        
                        let chosen_prob = probs.gather(1, chosen_index_tensor);
                        let log_prob = chosen_prob.log();

                        // REINFORCE formula: Loss = - (Reward * LogProb)
                        // This forces the network to increase Prob(Action) if Reward is high
                        let step_loss = log_prob.mul(reward_tensor.clone().unsqueeze::<2>()).neg();
                        loss_tensors.push(step_loss.reshape([1]));
                    }
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
            self.net = self.optimizer.step(self.learning_rate, self.net.clone(), grads);

            println!("✅ Model weights updated! The AI is a bit smarter now.");
        }
    }
}
