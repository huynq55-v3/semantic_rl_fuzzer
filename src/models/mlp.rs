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
