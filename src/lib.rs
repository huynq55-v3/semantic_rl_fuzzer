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
    pub log_probs: Vec<f32>,     // Neural network's confidence for calculating loss
    pub reward: f32,             // Total accumulated reward
    pub is_interesting: bool,    // Flag to keep this trajectory as a future mutation seed
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
        self.memory.choose_multiple(&mut rng, batch_size).cloned().collect()
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

    /// Takes the current state and valid action mask, returns the chosen action and its log probability.
    fn choose_action(&self, state: &Self::State, mask: &[bool]) -> (Self::Action, f32);
    
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
                log_probs: Vec::new(),
                reward: 0.0,
                is_interesting: false,
            };

            for _step in 0..self.max_steps_per_episode {
                let state = self.env.get_state();
                let mask = self.env.get_action_mask();

                let (action, log_prob) = self.agent.choose_action(&state, &mask);
                let result = self.env.step(&action);

                current_traj.states.push(state.clone());
                current_traj.actions.push(action.clone());
                current_traj.log_probs.push(log_prob);
                current_traj.reward += result.reward;

                if result.is_violated {
                    println!("🚨 [Episode {}] LOGIC BUG DETECTED! Terminating episode to save artifact.", episode);
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
                println!("🧠 [Episode {}] Batch training complete. Backpropagation applied!", episode);
            }
        }
    }
}

#[cfg(feature = "burn-backend")]
pub mod burn_helpers {
    // Nơi bạn định nghĩa sẵn các class bọc Burn Tensor 
    // để người dùng không phải hì hục code mạng Multi-head từ đầu.
}
