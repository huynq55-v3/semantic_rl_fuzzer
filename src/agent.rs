use crate::models::mlp::{MlpActor, MlpForward};
use crate::models::transformer::{TransformerActor, TransformerForward};
use crate::models::{ActorArchitecture, ForwardArchitecture, ModelArchitecture};
use burn::backend::{Autodiff, Wgpu};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand::distr::{weighted::WeightedIndex, Distribution, Uniform};
use rand::rng;
use rayon::prelude::*;

// 🌟 Gắn cứng Backend là Wgpu (GPU mặc định)
pub type FuzzBackend = Autodiff<Wgpu>;

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

pub trait ActionTranslator: Send + Sync + Clone {
    type TargetAction: Send + Sync + Clone + std::fmt::Debug;
    fn translate(&self, head_outputs: &[usize]) -> Self::TargetAction;
}

// 🌟 Loại bỏ Generic B
pub struct BurnAgent<T: ActionTranslator, ActorO, FwdO> {
    pub actor_net: ActorArchitecture<FuzzBackend>,
    pub forward_net: ForwardArchitecture<FuzzBackend>,
    pub actor_opt: ActorO,
    pub fwd_opt: FwdO,
    pub translator: T,
    pub learning_rate: f64,
    pub d_model: usize,
    pub device: <FuzzBackend as Backend>::Device,
    pub replay_buffer: ForwardReplayBuffer,
    pub intrinsic_weight: f32,
    pub entropy_coeff: f32,
    pub noise_floor: f32,
    pub batch_size: usize,
}

#[derive(Clone)]
pub struct BurnActor<T: ActionTranslator> {
    pub actor_net: ActorArchitecture<<FuzzBackend as AutodiffBackend>::InnerBackend>,
    pub translator: T,
    pub device: <<FuzzBackend as AutodiffBackend>::InnerBackend as Backend>::Device,
    pub noise_floor: f32,
}

impl<T: ActionTranslator> crate::core::FuzzActor for BurnActor<T> {
    type State = Vec<f32>;
    type Action = T::TargetAction;

    fn choose_action(
        &self,
        state: &Self::State,
        masks: &[Vec<bool>],
    ) -> (Self::Action, Vec<usize>, f32) {
        let state_tensor = Tensor::from_data(
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
                let mask_tensor = Tensor::<_, 2, Bool>::from_data(
                    TensorData::new(masks[i].clone(), [1, head_size]),
                    &self.device,
                );
                logits = logits.mask_fill(mask_tensor.bool_not(), -1e9);
            }

            let probs = burn::tensor::activation::softmax(logits, 1).clamp_min(1e-8);
            let probs_vec = probs.into_data().to_vec::<f32>().unwrap();
            let mut rng = rng();

            let dist_result = WeightedIndex::new(&probs_vec);
            let chosen_index = match dist_result {
                Ok(dist) => dist.sample(&mut rng),
                Err(_) => {
                    if !masks[i].is_empty() {
                        let valid: Vec<usize> = masks[i]
                            .iter()
                            .enumerate()
                            .filter(|(_, &v)| v)
                            .map(|(idx, _)| idx)
                            .collect();
                        if !valid.is_empty() {
                            valid[Uniform::new(0, valid.len()).unwrap().sample(&mut rng)]
                        } else {
                            0
                        }
                    } else {
                        Uniform::new(0, head_size).unwrap().sample(&mut rng)
                    }
                }
            };

            selected_indices.push(chosen_index);
            total_log_prob += probs_vec[chosen_index].max(1e-8).ln();
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
        if batch_size == 0 {
            return Vec::new();
        }
        let state_dim = states[0].len();

        let mut flattened_states = Vec::with_capacity(batch_size * state_dim);
        for s in states {
            flattened_states.extend_from_slice(s);
        }

        let state_tensor = Tensor::from_data(
            TensorData::new(flattened_states, [batch_size, state_dim]),
            &self.device,
        );

        let head_logits = self
            .actor_net
            .forward_with_floor(state_tensor, self.noise_floor);
        let num_heads = head_logits.len();

        let mut heads_probs_vecs = Vec::with_capacity(num_heads);
        let mut heads_sizes = Vec::with_capacity(num_heads);

        for (i, mut logits) in head_logits.into_iter().enumerate() {
            let head_size = logits.dims()[1];
            heads_sizes.push(head_size);

            let mut flat_mask = Vec::with_capacity(batch_size * head_size);
            for b in 0..batch_size {
                if masks_batch[b][i].is_empty() {
                    flat_mask.extend(std::iter::repeat(true).take(head_size));
                } else {
                    flat_mask.extend(masks_batch[b][i].iter().copied());
                }
            }

            let mask_tensor = Tensor::<_, 2, Bool>::from_data(
                TensorData::new(flat_mask, [batch_size, head_size]),
                &self.device,
            );
            logits = logits.mask_fill(mask_tensor.bool_not(), -1e9);

            let probs = burn::tensor::activation::softmax(logits, 1).clamp_min(1e-8);
            heads_probs_vecs.push(probs.into_data().to_vec::<f32>().unwrap());
        }

        let translator = &self.translator;
        (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut selected_indices = Vec::with_capacity(num_heads);
                let mut total_log_prob = 0.0;
                let mut rng = rng();

                for h in 0..num_heads {
                    let h_size = heads_sizes[h];
                    let start = b * h_size;
                    let end = start + h_size;
                    let probs_slice = &heads_probs_vecs[h][start..end];

                    let dist_result = WeightedIndex::new(probs_slice);
                    let chosen_index = match dist_result {
                        Ok(dist) => dist.sample(&mut rng),
                        Err(_) => {
                            if !masks_batch[b][h].is_empty() {
                                let valid: Vec<usize> = masks_batch[b][h]
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, &v)| v)
                                    .map(|(idx, _)| idx)
                                    .collect();
                                if !valid.is_empty() {
                                    valid[Uniform::new(0, valid.len()).unwrap().sample(&mut rng)]
                                } else {
                                    0
                                }
                            } else {
                                Uniform::new(0, h_size).unwrap().sample(&mut rng)
                            }
                        }
                    };

                    selected_indices.push(chosen_index);
                    total_log_prob += probs_slice[chosen_index].max(1e-8).ln();
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

impl<T: ActionTranslator, ActorO, FwdO> crate::core::NeuralAgent for BurnAgent<T, ActorO, FwdO>
where
    ActorO: Optimizer<ActorArchitecture<FuzzBackend>, FuzzBackend>,
    FwdO: Optimizer<ForwardArchitecture<FuzzBackend>, FuzzBackend>,
{
    type State = Vec<f32>;
    type Action = T::TargetAction;
    type Actor = BurnActor<T>;

    fn get_actor(&self) -> Self::Actor {
        BurnActor {
            actor_net: self.actor_net.valid(),
            translator: self.translator.clone(),
            device: self.device.clone(),
            noise_floor: self.noise_floor,
        }
    }

    fn learn_from_batch(
        &mut self,
        trajectories: &[crate::core::Trajectory<Self::State, Self::Action>],
    ) -> f32 {
        let head_sizes = self.actor_net.head_sizes();
        let total_action_dims: usize = head_sizes.iter().sum();

        let mut all_s = Vec::new();
        let mut all_sn = Vec::new();
        let mut all_ai_i64 = Vec::new();
        let mut all_ai_f32 = Vec::new();
        let mut all_rewards = Vec::new();

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

                self.replay_buffer.push(s_t, &one_hot, s_n);
            }
        }

        if all_s.is_empty() {
            return 0.0;
        }

        let total_steps = all_rewards.len();
        let state_dim = trajectories[0].states[0].len();

        let fwd_epochs = 5;
        let fwd_batch_size = self.batch_size.min(self.replay_buffer.memory.len());
        let mut avg_fwd_loss = 0.0;

        if self.replay_buffer.memory.len() >= fwd_batch_size {
            for _ in 0..fwd_epochs {
                let (b_s, b_a, b_sn) = self.replay_buffer.sample(fwd_batch_size);

                let s_tens = Tensor::<FuzzBackend, 2>::from_data(
                    TensorData::new(b_s, [fwd_batch_size, state_dim]),
                    &self.device,
                );
                let a_tens = Tensor::<FuzzBackend, 2>::from_data(
                    TensorData::new(b_a, [fwd_batch_size, total_action_dims]),
                    &self.device,
                );
                let sn_tens = Tensor::<FuzzBackend, 2>::from_data(
                    TensorData::new(b_sn, [fwd_batch_size, state_dim]),
                    &self.device,
                );

                let pred_sn = self.forward_net.forward_step(s_tens, a_tens);
                let fwd_loss = (pred_sn - sn_tens).powf_scalar(2.0).mean();

                avg_fwd_loss += fwd_loss.clone().into_data().to_vec::<f32>().unwrap()[0];

                let fwd_grads = fwd_loss.backward();
                let fwd_grads_params = GradientsParams::from_grads(fwd_grads, &self.forward_net);
                self.forward_net = self.fwd_opt.step(
                    self.learning_rate,
                    self.forward_net.clone(),
                    fwd_grads_params,
                );
            }
            avg_fwd_loss /= fwd_epochs as f32;
        }

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
            let mb_a_f32 = all_ai_f32[start * total_action_dims..end * total_action_dims].to_vec();
            let mb_rew = all_rewards[start..end].to_vec();

            let s_tens = Tensor::<FuzzBackend, 2>::from_data(
                TensorData::new(mb_s, [current_batch_size, state_dim]),
                &self.device,
            );
            let sn_tens = Tensor::<FuzzBackend, 2>::from_data(
                TensorData::new(mb_sn, [current_batch_size, state_dim]),
                &self.device,
            );
            let a_tens = Tensor::<FuzzBackend, 2>::from_data(
                TensorData::new(mb_a_f32, [current_batch_size, total_action_dims]),
                &self.device,
            );
            let ext_rew_tens = Tensor::<FuzzBackend, 1>::from_data(
                TensorData::new(mb_rew, [current_batch_size]),
                &self.device,
            );

            let pred_sn = self.forward_net.forward_step(s_tens.clone(), a_tens);
            let mse = (pred_sn - sn_tens).powf_scalar(2.0);
            let int_rewards = mse.mean_dim(1).reshape([current_batch_size]).detach();

            avg_curiosity += int_rewards
                .clone()
                .mean()
                .into_data()
                .to_vec::<f32>()
                .unwrap()[0];

            let scale_factor = (state_dim as f32 / 15.0).max(1.0);
            let intrinsic_weight = self.intrinsic_weight * scale_factor;
            let total_rewards = ext_rew_tens.add(int_rewards.mul_scalar(intrinsic_weight as f64));

            let all_head_logits = self.actor_net.forward_with_floor(s_tens, self.noise_floor);
            let mut actor_loss_sum = Tensor::<FuzzBackend, 1>::from_data([0.0], &self.device);

            for (h, logits) in all_head_logits.into_iter().enumerate() {
                let probs = burn::tensor::activation::softmax(logits.clone(), 1).clamp_min(1e-8);
                let log_probs_all = burn::tensor::activation::log_softmax(logits, 1);
                let entropy = probs.clone().mul(log_probs_all).sum_dim(1).neg().mean();

                let mb_a_i64: Vec<i64> = all_ai_i64
                    [start * head_sizes.len()..end * head_sizes.len()]
                    .iter()
                    .skip(h)
                    .step_by(head_sizes.len())
                    .copied()
                    .collect();

                let index_tensor = Tensor::<FuzzBackend, 2, burn::tensor::Int>::from_data(
                    TensorData::new(mb_a_i64, [current_batch_size, 1]),
                    &self.device,
                );
                let safe_probs = probs.gather(1, index_tensor).clamp_min(1e-8);
                let log_probs_selected = safe_probs.log().reshape([current_batch_size]);

                let policy_loss = log_probs_selected.mul(total_rewards.clone()).neg().mean();
                let head_loss = policy_loss.sub(entropy.mul_scalar(self.entropy_coeff as f64));
                actor_loss_sum = actor_loss_sum.add(head_loss);
            }

            avg_actor_loss += actor_loss_sum.clone().into_data().to_vec::<f32>().unwrap()[0];

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

        println!("[Batch] {} steps | Replay Mem: {}/{} | Int_μ: {:.4} | Act_Loss: {:.4} | Fwd_Loss: {:.4}", 
            total_steps, self.replay_buffer.memory.len(), self.replay_buffer.capacity, avg_curiosity, avg_actor_loss, avg_fwd_loss);

        avg_curiosity
    }
}

// 🌟 API Cực Kỳ Tối Giản
pub fn create_agent<T: ActionTranslator>(
    arch: ModelArchitecture,
    input_size: usize,
    d_model: usize,
    head_sizes: &[usize],
    learning_rate: f64,
    translator: T,
    intrinsic_weight: f32,
    entropy_coeff: f32,
    noise_floor: f32,
    batch_size: usize,
    buffer_capacity: usize,
) -> BurnAgent<
    T,
    impl Optimizer<ActorArchitecture<FuzzBackend>, FuzzBackend>,
    impl Optimizer<ForwardArchitecture<FuzzBackend>, FuzzBackend>,
> {
    // Tự động khởi tạo Device bên trong
    let device = <FuzzBackend as Backend>::Device::default();

    let actor_net = match arch {
        ModelArchitecture::Mlp => {
            ActorArchitecture::Mlp(MlpActor::new(&device, input_size, d_model, head_sizes))
        }
        ModelArchitecture::Transformer => ActorArchitecture::Transformer(TransformerActor::new(
            &device, input_size, d_model, head_sizes,
        )),
    };

    let total_action_dims: usize = head_sizes.iter().sum();

    let forward_net = match arch {
        ModelArchitecture::Mlp => ForwardArchitecture::Mlp(MlpForward::new(
            &device,
            input_size,
            total_action_dims,
            d_model,
        )),
        ModelArchitecture::Transformer => ForwardArchitecture::Transformer(
            TransformerForward::new(&device, input_size, total_action_dims, d_model),
        ),
    };

    BurnAgent {
        actor_net,
        forward_net,
        actor_opt: AdamConfig::new().init(),
        fwd_opt: AdamConfig::new().init(),
        translator,
        learning_rate,
        d_model,
        device,
        replay_buffer: ForwardReplayBuffer::new(buffer_capacity),
        intrinsic_weight,
        entropy_coeff,
        noise_floor,
        batch_size,
    }
}
