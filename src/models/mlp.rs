use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

use crate::models::LOGIT_NOISE_MULTIPLIER;

#[derive(Module, Debug)]
pub struct MlpActor<B: Backend> {
    pub shared_layer_1: Linear<B>,
    pub shared_layer_2: Linear<B>,
    pub heads: Vec<Linear<B>>,
    pub relu: Relu,
}

impl<B: Backend> MlpActor<B> {
    pub fn new(
        device: &B::Device,
        input_size: usize,
        d_model: usize,
        head_sizes: &[usize],
    ) -> Self {
        let shared_layer_1 = LinearConfig::new(input_size, d_model).init(device);
        let shared_layer_2 = LinearConfig::new(d_model, d_model).init(device);
        let mut heads = Vec::new();
        for &size in head_sizes {
            heads.push(LinearConfig::new(d_model, size).init(device));
        }
        Self {
            shared_layer_1,
            shared_layer_2,
            heads,
            relu: Relu::new(),
        }
    }

    pub fn forward_with_floor(&self, state: Tensor<B, 2>, floor: f32) -> Vec<Tensor<B, 2>> {
        let x = self.shared_layer_1.forward(state);
        let x = self.relu.forward(x);
        let x = self.shared_layer_2.forward(x);
        let shared_features = self.relu.forward(x);

        self.heads
            .iter()
            .map(|head| {
                let logits = head.forward(shared_features.clone());
                if floor > 0.0 {
                    let noise = Tensor::<B, 2>::random(
                        logits.dims(),
                        burn::tensor::Distribution::Normal(
                            0.0,
                            floor as f64 * LOGIT_NOISE_MULTIPLIER,
                        ),
                        &logits.device(),
                    );
                    logits.add(noise)
                } else {
                    logits
                }
            })
            .collect()
    }
}

#[derive(Module, Debug)]
pub struct MlpForward<B: Backend> {
    fc_1: Linear<B>,
    fc_2: Linear<B>,
    out: Linear<B>,
    relu: Relu,
}

impl<B: Backend> MlpForward<B> {
    pub fn new(device: &B::Device, state_dim: usize, num_heads: usize, d_model: usize) -> Self {
        let input_dim = state_dim + num_heads;
        Self {
            fc_1: LinearConfig::new(input_dim, d_model * 2).init(device),
            fc_2: LinearConfig::new(d_model * 2, d_model * 2).init(device),
            out: LinearConfig::new(d_model * 2, state_dim).init(device),
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
