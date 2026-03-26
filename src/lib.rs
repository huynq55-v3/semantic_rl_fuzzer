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
    fn get_action_mask(&self) -> Vec<Vec<bool>>;
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
    fn choose_action(
        &self,
        state: &Self::State,
        masks: &[Vec<bool>],
    ) -> (Self::Action, Vec<usize>, f32);
}

/// The learner side: owns the Autodiff graph, produces lightweight Actors.
/// Only requires `Send` (not `Sync`) since it is never shared across threads.
pub trait NeuralAgent: Send {
    type State;
    type Action;
    type Actor: FuzzActor<State = Self::State, Action = Self::Action>;

    /// Extract a thread-safe actor (no Autodiff) for parallel inference.
    fn get_actor(&self) -> Self::Actor;
    fn learn_from_batch(&mut self, trajectories: &[Trajectory<Self::State, Self::Action>]) -> f32;
    fn reset_forward_net(&mut self);
    fn get_curiosity_threshold(&self) -> f32;
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
                        let masks = local_env.get_action_mask();

                        // Use ACTOR (owned, not shared) for thread-safe inference
                        let (action, indices, log_prob) = actor.choose_action(&state, &masks);
                        let result = local_env.step(&action);

                        traj.states.push(state.clone());
                        traj.actions.push(action.clone());
                        traj.action_indices.push(indices);
                        traj.log_probs.push(log_prob);

                        // Push next_state so ICM has (S_t, S_{t+1}) pairs
                        traj.states.push(result.next_state);

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

            let current_curiosity = self.agent.learn_from_batch(&rollouts);

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

                if current_curiosity < self.agent.get_curiosity_threshold() && crashes_found == 0 {
                    self.agent.reset_forward_net();
                }

                // Fire callback so consuming code can analyze the batch
                on_log(iteration, &rollouts);
            }

            rollouts.clear();
        }

        drop(artifact_tx);
        let _ = writer_thread.join();
    }
}

#[cfg(feature = "burn-backend")]
pub mod burn_helpers {
    use burn::backend::Autodiff;
    use burn::module::{AutodiffModule, Param};
    use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
    use burn::nn::{Linear, LinearConfig, Relu};
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};
    use burn::prelude::*;
    use burn::tensor::backend::AutodiffBackend;
    use rand::distr::{weighted::WeightedIndex, Distribution, Uniform};
    use rand::rng;

    // ==========================================
    // TRÍ NHỚ DÀI HẠN (Replay Buffer) - Giữ nguyên
    // ==========================================
    #[derive(Clone)]
    pub struct Transition {
        pub state: Vec<f32>,
        pub action_one_hot: Vec<f32>,
        pub next_state: Vec<f32>,
    }

    pub struct ForwardReplayBuffer {
        pub capacity: usize,
        pub memory: Vec<Transition>,
        pub ptr: usize,
    }

    impl ForwardReplayBuffer {
        pub fn new(capacity: usize) -> Self {
            Self {
                capacity,
                memory: Vec::with_capacity(capacity),
                ptr: 0,
            }
        }

        pub fn push(&mut self, state: &[f32], action: &[f32], next_state: &[f32]) {
            let t = Transition {
                state: state.to_vec(),
                action_one_hot: action.to_vec(),
                next_state: next_state.to_vec(),
            };
            if self.memory.len() < self.capacity {
                self.memory.push(t);
            } else {
                self.memory[self.ptr] = t;
            }
            self.ptr = (self.ptr + 1) % self.capacity;
        }

        pub fn sample(&self, batch_size: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let mut rng = rng();
            let mut states = Vec::with_capacity(batch_size * self.memory[0].state.len());
            let mut actions = Vec::with_capacity(batch_size * self.memory[0].action_one_hot.len());
            let mut next_states = Vec::with_capacity(batch_size * self.memory[0].next_state.len());

            let dist = Uniform::new(0, self.memory.len()).unwrap();
            for _ in 0..batch_size {
                let idx = dist.sample(&mut rng);
                let t = &self.memory[idx];
                states.extend_from_slice(&t.state);
                actions.extend_from_slice(&t.action_one_hot);
                next_states.extend_from_slice(&t.next_state);
            }
            (states, actions, next_states)
        }
    }

    // ==========================================
    // DEEPMIND'S NOISY LINEAR (Sửa lại chuẩn)
    // ==========================================
    #[derive(Module, Debug)]
    pub struct NoisyLinear<B: Backend> {
        pub mu_w: Param<Tensor<B, 2>>,
        pub sig_w: Param<Tensor<B, 2>>,
        pub mu_b: Param<Tensor<B, 1>>,
        pub sig_b: Param<Tensor<B, 1>>,
        pub d_out: usize,
    }

    impl<B: Backend> NoisyLinear<B> {
        pub fn new(device: &B::Device, d_in: usize, d_out: usize) -> Self {
            let bound = (1.0 / d_in as f64).sqrt();
            let mu_w = Tensor::<B, 2>::random(
                [d_in, d_out],
                burn::tensor::Distribution::Uniform(-bound, bound),
                device,
            );
            let mu_b = Tensor::<B, 1>::random(
                [d_out],
                burn::tensor::Distribution::Uniform(-bound, bound),
                device,
            );
            let sig_init = 0.5 / (d_in as f32).sqrt();
            let sig_w = Tensor::<B, 2>::zeros([d_in, d_out], device).add_scalar(sig_init);
            let sig_b = Tensor::<B, 1>::zeros([d_out], device).add_scalar(sig_init);
            Self {
                mu_w: Param::from_tensor(mu_w),
                sig_w: Param::from_tensor(sig_w),
                mu_b: Param::from_tensor(mu_b),
                sig_b: Param::from_tensor(sig_b),
                d_out,
            }
        }

        // 🌟 NOISY LINEAR CHỈ TRẢ VỀ 1 TENSOR (Không phải Vec)
        pub fn forward_with_floor(&self, x: Tensor<B, 2>, floor: f32) -> Tensor<B, 2> {
            let [_, d_in] = x.dims();
            let eps_w = Tensor::<B, 2>::random(
                [d_in, self.d_out],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &x.device(),
            );
            let eps_b = Tensor::<B, 1>::random(
                [self.d_out],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &x.device(),
            );
            let sig_w = self.sig_w.val().clamp_min(floor as f64);
            let sig_b = self.sig_b.val().clamp_min(floor as f64);
            let weight = self.mu_w.val().add(sig_w.mul(eps_w));
            let bias = self.mu_b.val().add(sig_b.mul(eps_b));
            x.matmul(weight).add(bias.unsqueeze_dim(0))
        }
    }

    // ==========================================
    // 1. MINI TRANSFORMER ACTOR (Khối logic riêng biệt)
    // ==========================================
    #[derive(Module, Debug)]
    pub struct InvariantTransformerNet<B: Backend> {
        pub token_proj: Linear<B>,
        pub attention: MultiHeadAttention<B>,
        pub fc_out: Linear<B>,
        pub heads: Vec<NoisyLinear<B>>,
        pub relu: Relu,
    }

    impl<B: Backend> InvariantTransformerNet<B> {
        pub fn new(
            device: &B::Device,
            _input_size: usize,
            d_model: usize,
            head_sizes: &[usize],
        ) -> Self {
            let token_proj = LinearConfig::new(1, d_model).init(device);
            let attention = MultiHeadAttentionConfig::new(d_model, 4).init(device);
            let fc_out = LinearConfig::new(d_model, d_model).init(device);

            let mut heads = Vec::new();
            for &size in head_sizes {
                heads.push(NoisyLinear::new(device, d_model, size));
            }

            Self {
                token_proj,
                attention,
                fc_out,
                heads,
                relu: Relu::new(),
            }
        }

        pub fn forward_with_floor(&self, state: Tensor<B, 2>, floor: f32) -> Vec<Tensor<B, 2>> {
            let [batch_size, seq_len] = state.dims();
            let x_seq = state.reshape([batch_size, seq_len, 1]);

            let x_proj = self.token_proj.forward(x_seq);

            // 🌟 SỬA LỖI API BURN: Gói vào MhaInput
            let mha_input = MhaInput::self_attn(x_proj);
            let mha_output = self.attention.forward(mha_input);

            // 🌟 Lấy context tensor từ Output
            let x_attn = mha_output.context;

            // 1. Lấy thông số chiều thực tế của Tensor 3D (Không hardcode)
            let [batch_size, _seq_len, d_model] = x_attn.dims();

            // 2. Dùng RESHAPE thay vì SQUEEZE: Đảm bảo luôn ra chuẩn 2 chiều [Batch, d_model]
            let x_pooled = x_attn.mean_dim(1).reshape([batch_size, d_model]);

            let shared_features = self.relu.forward(self.fc_out.forward(x_pooled));

            // Map qua các heads (Action Types, Index 1, Index 2...)
            self.heads
                .iter()
                .map(|head| head.forward_with_floor(shared_features.clone(), floor))
                .collect()
        }
    }

    // ==========================================
    // 2. FORWARD NETWORK (Cũng có thể dùng MLP cho nhẹ, hoặc đổi sang MiniTransformer tương tự)
    // Ở đây giữ MLP để tính toán nhanh, vì nó chỉ dự đoán next_state
    // ==========================================
    #[derive(Module, Debug)]
    pub struct ForwardNet<B: Backend> {
        fc_1: Linear<B>,
        fc_2: Linear<B>,
        out: Linear<B>,
        relu: Relu,
    }

    impl<B: Backend> ForwardNet<B> {
        pub fn new(
            device: &B::Device,
            state_dim: usize,
            num_heads: usize,
            hidden_dim: usize,
        ) -> Self {
            let input_dim = state_dim + num_heads;
            Self {
                fc_1: LinearConfig::new(input_dim, hidden_dim).init(device),
                fc_2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
                out: LinearConfig::new(hidden_dim, state_dim).init(device),
                relu: Relu::new(),
            }
        }
        pub fn forward(&self, state: Tensor<B, 2>, action_indices: Tensor<B, 2>) -> Tensor<B, 2> {
            let x = Tensor::cat(vec![state, action_indices], 1);
            let x = self.relu.forward(self.fc_1.forward(x));
            let x = self.relu.forward(self.fc_2.forward(x));
            self.out.forward(x)
        }
    }

    pub trait ActionTranslator: Send + Sync + Clone {
        type TargetAction: Send + Sync + Clone + std::fmt::Debug;
        fn translate(&self, head_outputs: &[usize]) -> Self::TargetAction;
    }

    // ==========================================
    // TÁI CẤU TRÚC AGENT VỚI GENERIC BACKEND
    // ==========================================
    pub struct BurnAgent<B: AutodiffBackend, T: ActionTranslator, ActorO, FwdO> {
        pub actor_net: InvariantTransformerNet<B>,
        pub forward_net: ForwardNet<B>,
        pub actor_opt: ActorO,
        pub fwd_opt: FwdO,
        pub translator: T,
        pub learning_rate: f64,
        pub device: B::Device,
        pub replay_buffer: ForwardReplayBuffer,
        pub intrinsic_weight: f32,
        pub entropy_coeff: f32,
        pub noise_floor: f32,
        pub batch_size: usize,
        pub curiosity_threshold: f32,
    }

    #[derive(Clone)]
    pub struct BurnActor<B: Backend, T: ActionTranslator> {
        pub actor_net: InvariantTransformerNet<B>,
        pub translator: T,
        pub device: B::Device,
        pub noise_floor: f32,
    }

    impl<B: Backend, T: ActionTranslator> super::FuzzActor for BurnActor<B, T> {
        type State = Vec<f32>;
        type Action = T::TargetAction;

        fn choose_action(
            &self,
            state: &Self::State,
            masks: &[Vec<bool>],
        ) -> (Self::Action, Vec<usize>, f32) {
            let state_tensor = Tensor::<B, 2>::from_data(
                TensorData::new(state.clone(), [1, state.len()]),
                &self.device,
            );
            let head_logits = self
                .actor_net
                .forward_with_floor(state_tensor, self.noise_floor);

            let mut selected_indices = Vec::new();
            let mut total_log_prob = 0.0;

            for (i, mut logits) in head_logits.into_iter().enumerate() {
                let head_size = logits.dims()[1];
                if !masks[i].is_empty() {
                    let mask_tensor = Tensor::<B, 2, Bool>::from_data(
                        TensorData::new(masks[i].clone(), [1, head_size]),
                        &self.device,
                    );
                    logits = logits.mask_fill(mask_tensor.bool_not(), -1e9);
                }

                let probs = burn::tensor::activation::softmax(logits, 1);
                let probs_vec = probs.into_data().to_vec::<f32>().unwrap();
                let mut rng = rng();
                let chosen_index = WeightedIndex::new(&probs_vec).unwrap().sample(&mut rng);

                selected_indices.push(chosen_index);
                total_log_prob += probs_vec[chosen_index].ln();
            }
            (
                self.translator.translate(&selected_indices),
                selected_indices,
                total_log_prob,
            )
        }
    }

    impl<B: AutodiffBackend, T: ActionTranslator, ActorO, FwdO> super::NeuralAgent
        for BurnAgent<B, T, ActorO, FwdO>
    where
        ActorO: Optimizer<InvariantTransformerNet<B>, B>,
        FwdO: Optimizer<ForwardNet<B>, B>,
    {
        type State = Vec<f32>;
        type Action = T::TargetAction;
        type Actor = BurnActor<B::InnerBackend, T>;

        fn reset_forward_net(&mut self) {
            let head_sizes: Vec<usize> = self.actor_net.heads.iter().map(|h| h.d_out).collect();
            let total_action_dims: usize = head_sizes.iter().sum();

            // Lấy lại config cũ để build lại FwdNet (Giả sử hidden_dim của FwdNet trùng d_model của Actor cho gọn)
            let hidden_dim = self.actor_net.fc_out.weight.dims()[1];
            let input_size = self.forward_net.out.weight.dims()[1]; // out: [hidden -> state_dim]

            self.forward_net =
                ForwardNet::<B>::new(&self.device, input_size, total_action_dims, hidden_dim);
            self.replay_buffer.memory.clear();
            self.replay_buffer.ptr = 0;
            println!("🧠 [Hệ Thống] Đã Reset Nhà Tiên Tri - Thế giới trở nên mới lạ!");
        }

        fn get_actor(&self) -> Self::Actor {
            BurnActor {
                actor_net: self.actor_net.valid(),
                translator: self.translator.clone(),
                device: self.device.clone(),
                noise_floor: self.noise_floor,
            }
        }

        fn get_curiosity_threshold(&self) -> f32 {
            self.curiosity_threshold
        }

        fn learn_from_batch(
            &mut self,

            trajectories: &[super::Trajectory<Self::State, Self::Action>],
        ) -> f32 {
            let head_sizes: Vec<usize> = self.actor_net.heads.iter().map(|h| h.d_out).collect();

            let total_action_dims: usize = head_sizes.iter().sum();

            let mut all_s = Vec::new();

            let mut all_sn = Vec::new();

            let mut all_ai_i64 = Vec::new();

            let mut all_ai_f32 = Vec::new();

            let mut all_rewards = Vec::new();

            // 1. GOM DỮ LIỆU & BƠM VÀO KHO TRÍ NHỚ

            for t in trajectories.iter().filter(|t| !t.action_indices.is_empty()) {
                for i in 0..t.action_indices.len() {
                    let s_t = &t.states[2 * i];

                    let s_n = &t.states[2 * i + 1];

                    all_s.extend_from_slice(s_t);

                    all_sn.extend_from_slice(s_n);

                    all_ai_i64.extend(t.action_indices[i].iter().map(|&idx| idx as i64));

                    let mut one_hot = vec![0.0f32; total_action_dims];

                    let mut offset = 0;

                    for (h, &val) in t.action_indices[i].iter().enumerate() {
                        if val < head_sizes[h] {
                            one_hot[offset + val] = 1.0;
                        }

                        offset += head_sizes[h];
                    }

                    all_ai_f32.extend(one_hot.clone());

                    all_rewards.push(t.reward);

                    // Đẩy dữ liệu vào kho lưu trữ Dài hạn

                    self.replay_buffer.push(s_t, &one_hot, s_n);
                }
            }

            if all_s.is_empty() {
                return 0.0;
            }

            let total_steps = all_rewards.len();

            let state_dim = trajectories[0].states[0].len();

            // ==========================================

            // PHASE 1: TRAIN FORWARD NET (OFF-POLICY MINI-BATCH)

            // Ép Tiên Tri ôn lại bài cũ 10 lần (Epochs) để không bao giờ quên

            // ==========================================

            let fwd_epochs = 5;

            let fwd_batch_size = self.batch_size.min(self.replay_buffer.memory.len());

            let mut avg_fwd_loss = 0.0;

            if self.replay_buffer.memory.len() >= fwd_batch_size {
                for _ in 0..fwd_epochs {
                    let (b_s, b_a, b_sn) = self.replay_buffer.sample(fwd_batch_size);

                    let s_tens = Tensor::<B, 2>::from_data(
                        TensorData::new(b_s, [fwd_batch_size, state_dim]),
                        &self.device,
                    );

                    let a_tens = Tensor::<B, 2>::from_data(
                        TensorData::new(b_a, [fwd_batch_size, total_action_dims]),
                        &self.device,
                    );

                    let sn_tens = Tensor::<B, 2>::from_data(
                        TensorData::new(b_sn, [fwd_batch_size, state_dim]),
                        &self.device,
                    );

                    let pred_sn = self.forward_net.forward(s_tens, a_tens);

                    let fwd_loss = (pred_sn - sn_tens).powf_scalar(2.0).mean();

                    avg_fwd_loss += fwd_loss.clone().into_data().to_vec::<f32>().unwrap()[0];

                    let fwd_grads = fwd_loss.backward();

                    let fwd_grads_params =
                        GradientsParams::from_grads(fwd_grads, &self.forward_net);

                    // Dùng Optimizer riêng cho Tiên Tri

                    self.forward_net = self.fwd_opt.step(
                        self.learning_rate,
                        self.forward_net.clone(),
                        fwd_grads_params,
                    );
                }

                avg_fwd_loss /= fwd_epochs as f32;
            }

            // ==========================================

            // PHASE 2: TRAIN ACTOR (ON-POLICY MINI-BATCH)

            // Băm nhỏ 51,200 steps ra thành từng mẻ 1024 để tiêu hóa

            // ==========================================

            let actor_batch_size = self.batch_size;

            let num_batches = (total_steps + actor_batch_size - 1) / actor_batch_size;

            let mut avg_actor_loss = 0.0;

            let mut avg_curiosity = 0.0;

            for b in 0..num_batches {
                let start = b * actor_batch_size;

                let end = (start + actor_batch_size).min(total_steps);

                let current_batch_size = end - start;

                let mb_s = all_s[start * state_dim..end * state_dim].to_vec();

                let mb_sn = all_sn[start * state_dim..end * state_dim].to_vec();

                let mb_a_f32 =
                    all_ai_f32[start * total_action_dims..end * total_action_dims].to_vec();

                let mb_rew = all_rewards[start..end].to_vec();

                let s_tens = Tensor::<B, 2>::from_data(
                    TensorData::new(mb_s, [current_batch_size, state_dim]),
                    &self.device,
                );

                let sn_tens = Tensor::<B, 2>::from_data(
                    TensorData::new(mb_sn, [current_batch_size, state_dim]),
                    &self.device,
                );

                let a_tens = Tensor::<B, 2>::from_data(
                    TensorData::new(mb_a_f32, [current_batch_size, total_action_dims]),
                    &self.device,
                );

                let ext_rew_tens = Tensor::<B, 1>::from_data(
                    TensorData::new(mb_rew, [current_batch_size]),
                    &self.device,
                );

                // Dùng ForwardNet VỪA ĐƯỢC CẬP NHẬT để tính điểm tò mò (Không Backprop cho FwdNet ở đây)

                let pred_sn = self.forward_net.forward(s_tens.clone(), a_tens);

                let mse = (pred_sn - sn_tens).powf_scalar(2.0);

                // Detach để gradient không lan sang Forward Net

                let int_rewards = mse.mean_dim(1).reshape([current_batch_size]).detach();

                avg_curiosity += int_rewards
                    .clone()
                    .mean()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()[0];

                let scale_factor = (state_dim as f32 / 15.0).max(1.0);

                let intrinsic_weight = self.intrinsic_weight * scale_factor;

                let total_rewards =
                    ext_rew_tens.add(int_rewards.mul_scalar(intrinsic_weight as f64));

                let all_head_logits = self.actor_net.forward_with_floor(s_tens, self.noise_floor);

                let mut actor_loss_sum = Tensor::<B, 1>::from_data([0.0], &self.device);

                let num_heads = head_sizes.len();

                for (h, logits) in all_head_logits.into_iter().enumerate() {
                    let probs = burn::tensor::activation::softmax(logits.clone(), 1);

                    let log_probs_all = burn::tensor::activation::log_softmax(logits, 1);

                    // 1. Calculate Entropy: H = -sum(p * log(p))

                    let entropy = probs.clone().mul(log_probs_all).sum_dim(1).neg().mean();

                    // 2. Calculate Policy Loss

                    let mb_a_i64: Vec<i64> = all_ai_i64[start * num_heads..end * num_heads]
                        .iter()
                        .skip(h)
                        .step_by(num_heads)
                        .copied()
                        .collect();

                    let index_tensor = Tensor::<B, 2, Int>::from_data(
                        TensorData::new(mb_a_i64, [current_batch_size, 1]),
                        &self.device,
                    );

                    let safe_probs = probs.gather(1, index_tensor).clamp_min(1e-8);

                    let log_probs_selected = safe_probs.log().reshape([current_batch_size]);

                    let policy_loss = log_probs_selected.mul(total_rewards.clone()).neg().mean();

                    // 3. Combined Loss: Total_Head_Loss = Policy_Loss - (beta * Entropy)

                    let head_loss = policy_loss.sub(entropy.mul_scalar(self.entropy_coeff as f64));

                    actor_loss_sum = actor_loss_sum.add(head_loss);
                }

                avg_actor_loss += actor_loss_sum.clone().into_data().to_vec::<f32>().unwrap()[0];

                // Cập nhật ĐỘC LẬP cho Actor Net

                let actor_grads = actor_loss_sum.backward();

                let actor_grads_params = GradientsParams::from_grads(actor_grads, &self.actor_net);

                self.actor_net = self.actor_opt.step(
                    self.learning_rate,
                    self.actor_net.clone(),
                    actor_grads_params,
                );
            }

            avg_actor_loss /= num_batches as f32;

            avg_curiosity /= num_batches as f32;

            println!(

                "🔥 Batch ({} steps) | Replay Mem: {}/{} | Int_μ: {:.4} | Act_Loss: {:.4} | Fwd_Loss: {:.4}",

                total_steps, self.replay_buffer.memory.len(), self.replay_buffer.capacity, avg_curiosity, avg_actor_loss, avg_fwd_loss

            );

            avg_curiosity
        }
    }

    // ==========================================
    // API SÁNG TẠO AGENT MỚI (Hỗ trợ GPU/CPU)
    // ==========================================
    pub fn create_agent<B: AutodiffBackend, T: ActionTranslator>(
        device: &B::Device, // Truyền thiết bị vào (CPU / WGPU / CUDA)
        input_size: usize,
        d_model: usize, // Thay vì hidden_size to đùng, giờ chỉ cần d_model (ví dụ 64 hoặc 128)
        head_sizes: &[usize],
        learning_rate: f64,
        translator: T,
        intrinsic_weight: f32,
        entropy_coeff: f32,
        noise_floor: f32,
        curiosity_threshold: f32,
        batch_size: usize,
        buffer_capacity: usize,
    ) -> BurnAgent<
        B,
        T,
        impl Optimizer<InvariantTransformerNet<B>, B>,
        impl Optimizer<ForwardNet<B>, B>,
    > {
        let actor_net = InvariantTransformerNet::<B>::new(device, input_size, d_model, head_sizes);

        let total_action_dims: usize = head_sizes.iter().sum();
        let forward_net = ForwardNet::<B>::new(device, input_size, total_action_dims, d_model * 2);

        let actor_opt = AdamConfig::new().init();
        let fwd_opt = AdamConfig::new().init();

        BurnAgent {
            actor_net,
            forward_net,
            actor_opt,
            fwd_opt,
            translator,
            learning_rate,
            device: device.clone(),
            replay_buffer: ForwardReplayBuffer::new(buffer_capacity),
            intrinsic_weight,
            entropy_coeff,
            noise_floor,
            batch_size,
            curiosity_threshold,
        }
    }
}
