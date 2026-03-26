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

    fn choose_batch_action(
        &self,
        states: &[Self::State],
        masks_batch: &[Vec<Vec<bool>>],
    ) -> Vec<(Self::Action, Vec<usize>, f32)>;
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

            // Extract template actor (thread-safe, no Autodiff)
            let actor = self.agent.get_actor();
            let oracle_ref = &self.oracle;
            let num_envs = self.config.num_envs;
            let max_steps = self.config.max_steps_per_episode;

            // 1. Khởi tạo lô Environment
            let mut envs: Vec<E> = vec![self.base_env.clone(); num_envs];
            for env in envs.iter_mut() {
                env.reset();
            }

            // 2. Khởi tạo lô Trajectory ghi nhật ký
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

            // ==========================================
            // 🌟 VÒNG LẶP ĐỒNG BỘ: Tất cả Environment tiến 1 bước cùng lúc
            // ==========================================
            for _step in 0..max_steps {
                // Bước 1: Thu thập State và Mask cực nhanh bằng Rayon
                // Đây chính là state TRƯỚC KHI hành động
                let current_states: Vec<E::State> =
                    envs.par_iter().map(|e| e.get_state()).collect();

                let current_masks: Vec<Vec<Vec<bool>>> =
                    envs.par_iter().map(|e| e.get_action_mask()).collect();

                // Bước 2: BATCH INFERENCE - Gửi toàn bộ lên GPU trong 1 lệnh duy nhất!
                let batch_results = actor.choose_batch_action(&current_states, &current_masks);

                // Bước 3: Áp dụng Action vào các Environment song song (Dùng luôn current_states đã lấy)
                let step_results: Vec<_> = envs
                    .par_iter_mut()
                    .zip(current_states.into_par_iter()) // 🌟 ZIP LUN STATE Ở BƯỚC 1 VÀO ĐÂY
                    .zip(batch_results.into_par_iter())
                    .zip(active_mask.par_iter())
                    .map(
                        |(((env, state_before), (action, indices, log_prob)), &is_active)| {
                            if !is_active {
                                return None;
                            }

                            // 🌟 KHÔNG GỌI `env.get_state()` Ở ĐÂY NỮA, DÙNG LUÔN `state_before`
                            let result = env.step(&action);
                            let status = oracle_ref.judge(env, result.is_invalid);

                            Some((
                                state_before, // Đã an toàn và đồng bộ tuyệt đối
                                action,
                                indices,
                                log_prob,
                                result.next_state,
                                status,
                            ))
                        },
                    )
                    .collect();

                // Bước 4: Cập nhật Trajectories và kiểm tra điều kiện dừng
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

                // Nếu tất cả môi trường đều sập/hoàn thành, kết thúc sớm vòng lặp steps
                if !any_active {
                    break;
                }
            }

            global_episodes += num_envs;

            // ==========================================
            // XỬ LÝ KẾT QUẢ & TRAIN
            // ==========================================
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

                    // Tái phát hiện crash để xuất Artifacts
                    if traj.reward <= -1.0 || traj.states.is_empty() {
                        crashes_found += 1;
                        let filename = format!("artifacts/bug_iter_{}_env_{}.txt", iteration, i);
                        let content = format!("Action sequence:\n{:#?}", traj.actions);
                        let _ = artifact_tx.send((filename, content));
                    }
                }
            }

            // Gửi cả lô cho Actor học
            let current_curiosity = self.agent.learn_from_batch(&rollouts);

            // ==========================================
            // LOGGING & CALLBACK
            // ==========================================
            if iteration % self.config.log_interval == 0 {
                let avg_reward = total_batch_reward / num_envs as f32;
                let elapsed = start_time.elapsed().as_secs_f64();

                // Tính toán FPS dựa trên số bước thực tế đã chạy
                let fps = total_steps_taken as f64 / elapsed;

                println!(
                    "📊 [Iter {} | Ep {}] Avg Reward: {:.2} | Crashes: {} | Speed: {:.0} steps/s",
                    iteration, global_episodes, avg_reward, crashes_found, fps
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
    use rayon::prelude::*;

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
    // 1. ACTOR NETWORK (Loại bỏ Noisy, Dùng Linear Thuần)
    // ==========================================
    #[derive(Module, Debug)]
    pub struct MultiHeadNet<B: Backend> {
        shared_layer_1: Linear<B>,
        shared_layer_2: Linear<B>,
        pub heads: Vec<Linear<B>>, // 🌟 Đổi về Linear thường
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
                heads.push(LinearConfig::new(hidden_size, size).init(device)); // 🌟 Dùng LinearConfig
            }
            Self {
                shared_layer_1,
                shared_layer_2,
                heads,
                relu: Relu::new(),
            }
        }

        pub fn forward_with_floor(&self, state: Tensor<B, 2>, _floor: f32) -> Vec<Tensor<B, 2>> {
            // Không cần floor nữa vì không có nhiễu
            let x = self.shared_layer_1.forward(state);
            let x = self.relu.forward(x);
            let x = self.shared_layer_2.forward(x);
            let shared_features = self.relu.forward(x);
            self.heads
                .iter()
                .map(|head| head.forward(shared_features.clone())) // 🌟 Gọi forward bình thường
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
        pub actor_net: MultiHeadNet<B>,
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
        pub actor_net: MultiHeadNet<B>,
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

        fn choose_batch_action(
            &self,
            states: &[Self::State],
            masks_batch: &[Vec<Vec<bool>>],
        ) -> Vec<(Self::Action, Vec<usize>, f32)> {
            let batch_size = states.len();
            let state_dim = states[0].len();

            // 1. Đẩy 1 cục lên GPU
            let mut flattened_states = Vec::with_capacity(batch_size * state_dim);
            for s in states {
                flattened_states.extend_from_slice(s);
            }

            let state_tensor = Tensor::<B, 2>::from_data(
                TensorData::new(flattened_states, [batch_size, state_dim]),
                &self.device,
            );

            // 2. Chạy qua mạng nơ-ron
            let head_logits = self
                .actor_net
                .forward_with_floor(state_tensor, self.noise_floor);
            let num_heads = head_logits.len();

            // 3. Xử lý Softmax và kéo về CPU theo từng Head
            let mut heads_probs_vecs = Vec::with_capacity(num_heads);
            let mut heads_sizes = Vec::with_capacity(num_heads);

            for (i, mut logits) in head_logits.into_iter().enumerate() {
                let head_size = logits.dims()[1];
                heads_sizes.push(head_size);

                // Áp Mask y hệt choose_action (nhưng cho cả Batch)
                let mut flat_mask = Vec::with_capacity(batch_size * head_size);
                for b in 0..batch_size {
                    if masks_batch[b][i].is_empty() {
                        flat_mask.extend(std::iter::repeat(true).take(head_size));
                    } else {
                        flat_mask.extend(masks_batch[b][i].iter().copied());
                    }
                }

                let mask_tensor = Tensor::<B, 2, Bool>::from_data(
                    TensorData::new(flat_mask, [batch_size, head_size]),
                    &self.device,
                );
                logits = logits.mask_fill(mask_tensor.bool_not(), -1e9);

                // Softmax và kéo về RAM
                let probs = burn::tensor::activation::softmax(logits, 1);
                heads_probs_vecs.push(probs.into_data().to_vec::<f32>().unwrap());
            }

            // 4. "Bốc thăm" song song bằng Rayon - Logic copy-paste từ bản đơn lẻ
            let translator = &self.translator;
            (0..batch_size)
                .into_par_iter()
                .map(|b| {
                    let mut selected_indices = Vec::with_capacity(num_heads);
                    let mut total_log_prob = 0.0;
                    let mut rng = rand::rng();

                    for h in 0..num_heads {
                        let h_size = heads_sizes[h];
                        let start = b * h_size;
                        let end = start + h_size;
                        let probs_slice = &heads_probs_vecs[h][start..end];

                        // 🌟 ĐÂY RỒI: Y hệt bản cũ của ông, dùng unwrap() thẳng mặt
                        let dist = WeightedIndex::new(probs_slice).unwrap();
                        let chosen_index = dist.sample(&mut rng);

                        selected_indices.push(chosen_index);
                        total_log_prob += probs_slice[chosen_index].ln();
                    }

                    (
                        translator.translate(&selected_indices),
                        selected_indices,
                        total_log_prob,
                    )
                })
                .collect()
        }
    }

    impl<B: AutodiffBackend, T: ActionTranslator, ActorO, FwdO> super::NeuralAgent
        for BurnAgent<B, T, ActorO, FwdO>
    where
        ActorO: Optimizer<MultiHeadNet<B>, B>,
        FwdO: Optimizer<ForwardNet<B>, B>,
    {
        type State = Vec<f32>;
        type Action = T::TargetAction;
        type Actor = BurnActor<B::InnerBackend, T>;

        fn reset_forward_net(&mut self) {
            // 1. Lấy thông số từ actor_net để giữ nguyên cấu trúc
            let weight_dims = self.actor_net.shared_layer_1.weight.dims();
            let input_size = weight_dims[0];
            let hidden_size = weight_dims[1];

            let head_sizes: Vec<usize> = self
                .actor_net
                .heads
                .iter()
                .map(|h| h.weight.dims()[1])
                .collect();
            let total_action_dims: usize = head_sizes.iter().sum();

            // 2. Khởi tạo mạng Forward mới tinh (Xóa sạch ký ức cũ)
            self.forward_net =
                ForwardNet::<B>::new(&self.device, input_size, total_action_dims, hidden_size);

            // 🌟 LƯU Ý: Không gán lại self.fwd_opt ở đây để tránh lỗi Mismatched Types.
            // Optimizer cũ sẽ tiếp tục làm việc với các trọng số mới của forward_net.

            // 3. Xóa sạch bộ nhớ Replay Buffer để xóa bỏ "định kiến" cũ
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
            let head_sizes: Vec<usize> = self
                .actor_net
                .heads
                .iter()
                .map(|h| h.weight.dims()[1])
                .collect();

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
    ) -> BurnAgent<B, T, impl Optimizer<MultiHeadNet<B>, B>, impl Optimizer<ForwardNet<B>, B>> {
        let actor_net = MultiHeadNet::<B>::new(device, input_size, d_model, head_sizes);

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
