use crate::models::mlp::MlpActor;
use crate::models::transformer::TransformerActor; // 🌟
use crate::models::{ActorArchitecture, ModelArchitecture};
use burn::backend::{Autodiff, Wgpu};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand::distr::{weighted::WeightedIndex, Distribution, Uniform};
use rand::rng;
use rayon::prelude::*;

pub type FuzzBackend = Autodiff<Wgpu>;

pub trait ActionTranslator: Send + Sync + Clone {
    type TargetAction: Send + Sync + Clone + std::fmt::Debug;
    fn translate(&self, head_outputs: &[usize]) -> Self::TargetAction;
}

pub struct BurnAgent<T: ActionTranslator, ActorO> {
    pub actor_net: ActorArchitecture<FuzzBackend>,
    pub actor_opt: ActorO,
    pub translator: T,
    pub learning_rate: f64,
    pub d_model: usize,
    pub device: <FuzzBackend as Backend>::Device,
    pub entropy_coeff: f32,
    pub noise_floor: f32,
    pub batch_size: usize,
    pub seq_len: usize, // 🌟 BỘ NHỚ THỜI GIAN
}

#[derive(Clone)]
pub struct BurnActor<T: ActionTranslator> {
    pub actor_net: ActorArchitecture<<FuzzBackend as AutodiffBackend>::InnerBackend>,
    pub translator: T,
    pub device: <<FuzzBackend as AutodiffBackend>::InnerBackend as Backend>::Device,
    pub noise_floor: f32,
    pub seq_len: usize, // 🌟
}

impl<T: ActionTranslator> crate::core::FuzzActor for BurnActor<T> {
    type State = Vec<f32>;
    type Action = T::TargetAction;

    fn choose_action(
        &self,
        state_history: &[Self::State],
        masks: &[Vec<bool>],
    ) -> (Self::Action, Vec<usize>, f32) {
        self.choose_batch_action(&[state_history.to_vec()], &[masks.to_vec()])
            .pop()
            .unwrap()
    }

    fn choose_batch_action(
        &self,
        state_histories: &[Vec<Self::State>],
        masks_batch: &[Vec<Vec<bool>>],
    ) -> Vec<(Self::Action, Vec<usize>, f32)> {
        let batch_size = state_histories.len();
        if batch_size == 0 {
            return Vec::new();
        }
        let state_dim = state_histories[0][0].len();

        // 🌟 NÉN CHUỖI: BÙ SỐ 0 NẾU THIẾU BƯỚC, CẮT ĐUÔI NẾU THỪA BƯỚC
        let mut flattened_sequences = Vec::with_capacity(batch_size * self.seq_len * state_dim);

        for history in state_histories {
            let mut padded = vec![0.0f32; self.seq_len * state_dim];
            let take_len = history.len().min(self.seq_len);
            let start_idx = self.seq_len - take_len; // Left padding
            let hist_start = history.len() - take_len;

            for i in 0..take_len {
                let offset = (start_idx + i) * state_dim;
                padded[offset..offset + state_dim].copy_from_slice(&history[hist_start + i]);
            }
            flattened_sequences.extend(padded);
        }

        let sequence_tensor = Tensor::from_data(
            TensorData::new(flattened_sequences, [batch_size, self.seq_len, state_dim]),
            &self.device,
        );

        let head_logits = self
            .actor_net
            .forward_with_floor(sequence_tensor, self.noise_floor);
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

impl<T: ActionTranslator, ActorO> crate::core::NeuralAgent for BurnAgent<T, ActorO>
where
    ActorO: Optimizer<ActorArchitecture<FuzzBackend>, FuzzBackend>,
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
            seq_len: self.seq_len, // 🌟
        }
    }

    fn learn_from_batch(
        &mut self,
        trajectories: &[crate::core::Trajectory<Self::State, Self::Action>],
    ) {
        let head_sizes = self.actor_net.head_sizes();

        let mut all_s_seq = Vec::new(); // LƯU CẢ CHUỖI CHO MỖI SAMPLE
        let mut all_ai_i64 = Vec::new();
        let mut all_rewards = Vec::new();
        let mut all_masks_by_head: Vec<Vec<bool>> = vec![Vec::new(); head_sizes.len()];

        let state_dim = if !trajectories.is_empty() && !trajectories[0].states.is_empty() {
            trajectories[0].states[0].len()
        } else {
            return;
        };

        for t in trajectories.iter().filter(|t| !t.action_indices.is_empty()) {
            for i in 0..t.action_indices.len() {
                // 🌟 Lấy lịch sử từ bước 0 đến bước thứ i
                let current_hist = &t.states[0..=i];
                let mut padded_history = vec![0.0; self.seq_len * state_dim];

                let take_len = current_hist.len().min(self.seq_len);
                let start_idx = self.seq_len - take_len;
                let hist_start = current_hist.len() - take_len;

                for step in 0..take_len {
                    let offset = (start_idx + step) * state_dim;
                    padded_history[offset..offset + state_dim]
                        .copy_from_slice(&current_hist[hist_start + step]);
                }

                all_s_seq.extend(padded_history);
                all_ai_i64.extend(t.action_indices[i].iter().map(|&idx| idx as i64));
                all_rewards.push(t.reward);

                for h in 0..head_sizes.len() {
                    if t.masks[i][h].is_empty() {
                        all_masks_by_head[h].extend(std::iter::repeat(true).take(head_sizes[h]));
                    } else {
                        all_masks_by_head[h].extend(&t.masks[i][h]);
                    }
                }
            }
        }

        if all_rewards.is_empty() {
            return;
        }

        let total_steps = all_rewards.len();
        let num_batches = (total_steps + self.batch_size - 1) / self.batch_size;
        let mut avg_actor_loss = 0.0;
        let mut avg_entropy = 0.0;

        for b in 0..num_batches {
            let start = b * self.batch_size;
            let end = (start + self.batch_size).min(total_steps);
            let current_batch_size = end - start;

            let mb_s_seq = all_s_seq
                [start * self.seq_len * state_dim..end * self.seq_len * state_dim]
                .to_vec();
            let mb_rew = all_rewards[start..end].to_vec();

            let s_tens = Tensor::<FuzzBackend, 3>::from_data(
                TensorData::new(mb_s_seq, [current_batch_size, self.seq_len, state_dim]),
                &self.device,
            );

            let ext_rew_tens = Tensor::<FuzzBackend, 1>::from_data(
                TensorData::new(mb_rew, [current_batch_size]),
                &self.device,
            );

            let mean_reward = ext_rew_tens.clone().mean();
            let advantage = ext_rew_tens.sub(mean_reward).detach();

            let all_head_logits = self.actor_net.forward_with_floor(s_tens, self.noise_floor);
            let mut actor_loss_sum = Tensor::<FuzzBackend, 1>::from_data([0.0], &self.device);

            for (h, mut logits) in all_head_logits.into_iter().enumerate() {
                let h_size = head_sizes[h];
                let mb_mask = &all_masks_by_head[h][start * h_size..end * h_size];

                let mask_tensor = Tensor::<FuzzBackend, 2, Bool>::from_data(
                    TensorData::new(mb_mask.to_vec(), [current_batch_size, h_size]),
                    &self.device,
                );

                logits = logits.mask_fill(mask_tensor.clone().bool_not(), -1e9);

                let probs = burn::tensor::activation::softmax(logits.clone(), 1).clamp_min(1e-8);
                let log_probs_all = burn::tensor::activation::log_softmax(logits, 1);

                let entropy_elements = probs.clone().mul(log_probs_all);
                let valid_entropy = entropy_elements.mask_fill(mask_tensor.bool_not(), 0.0);
                let entropy = valid_entropy.sum_dim(1).neg().mean();

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

                let policy_loss = log_probs_selected.mul(advantage.clone()).neg().mean();
                let head_loss =
                    policy_loss.sub(entropy.clone().mul_scalar(self.entropy_coeff as f64));
                actor_loss_sum = actor_loss_sum.add(head_loss);

                avg_entropy +=
                    entropy.into_data().to_vec::<f32>().unwrap()[0] / (head_sizes.len() as f32);
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
        avg_entropy /= num_batches as f32;

        println!(
            "🔥 [Actor] {:>5} steps học | Loss: {:>7.4} | Entropy: {:>5.4}",
            total_steps, avg_actor_loss, avg_entropy
        );
    }
}

pub fn create_agent<T: ActionTranslator>(
    arch: ModelArchitecture,
    input_size: usize,
    d_model: usize,
    head_sizes: &[usize],
    learning_rate: f64,
    translator: T,
    entropy_coeff: f32,
    noise_floor: f32,
    batch_size: usize,
    seq_len: usize, // 🌟 TRUYỀN PARAMETER BỘ NHỚ
) -> BurnAgent<T, impl Optimizer<ActorArchitecture<FuzzBackend>, FuzzBackend>> {
    let device = <FuzzBackend as Backend>::Device::default();

    let actor_net = match arch {
        ModelArchitecture::Mlp => {
            // MLP nhận mảng đã bị làm bẹp: Số chiều State x Chiều dài chuỗi
            ActorArchitecture::Mlp(MlpActor::new(
                &device,
                input_size * seq_len,
                d_model,
                head_sizes,
            ))
        }
        ModelArchitecture::Transformer => ActorArchitecture::Transformer(TransformerActor::new(
            &device, input_size, d_model, head_sizes, seq_len,
        )),
    };

    BurnAgent {
        actor_net,
        actor_opt: AdamConfig::new().init(),
        translator,
        learning_rate,
        d_model,
        device,
        entropy_coeff,
        noise_floor,
        batch_size,
        seq_len, // 🌟
    }
}
