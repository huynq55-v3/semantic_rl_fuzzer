use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

use crate::models::LOGIT_NOISE_MULTIPLIER;

#[derive(Module, Debug)]
pub struct TransformerActor<B: Backend> {
    pub embedding: Linear<B>,
    pub attention: MultiHeadAttention<B>,
    pub heads: Vec<Linear<B>>,
    pub relu: Relu,
}

impl<B: Backend> TransformerActor<B> {
    pub fn new(
        device: &B::Device,
        input_size: usize,
        d_model: usize,
        head_sizes: &[usize],
    ) -> Self {
        let embedding = LinearConfig::new(input_size, d_model).init(device);
        let attention = MultiHeadAttentionConfig::new(d_model, 4).init(device);
        let mut heads = Vec::new();
        for &size in head_sizes {
            heads.push(LinearConfig::new(d_model, size).init(device));
        }
        Self {
            embedding,
            attention,
            heads,
            relu: Relu::new(),
        }
    }

    pub fn forward_with_floor(&self, state: Tensor<B, 2>, floor: f32) -> Vec<Tensor<B, 2>> {
        let x = self.relu.forward(self.embedding.forward(state));

        // Add sequence dimension: [batch, d_model] -> [batch, 1, d_model]
        let x = x.unsqueeze_dim(1);
        let mha_input = MhaInput::self_attn(x);
        let x = self.attention.forward(mha_input).context;

        // Squeeze back to 2D: [batch, 1, d_model] -> [batch, d_model]
        let shared_features = x.squeeze::<2>();

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
pub struct TransformerForward<B: Backend> {
    embedding: Linear<B>,
    attention: MultiHeadAttention<B>,
    out: Linear<B>,
    relu: Relu,
}

impl<B: Backend> TransformerForward<B> {
    pub fn new(device: &B::Device, state_dim: usize, num_heads: usize, d_model: usize) -> Self {
        let input_dim = state_dim + num_heads;
        Self {
            embedding: LinearConfig::new(input_dim, d_model * 2).init(device),
            attention: MultiHeadAttentionConfig::new(d_model * 2, 4).init(device),
            out: LinearConfig::new(d_model * 2, state_dim).init(device),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, state: Tensor<B, 2>, action_indices: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = Tensor::cat(vec![state, action_indices], 1);
        let x = self.relu.forward(self.embedding.forward(x));

        let x = x.unsqueeze_dim(1);
        let mha_input = MhaInput::self_attn(x);
        let x = self.attention.forward(mha_input).context;

        let x = x.squeeze::<2>();
        self.out.forward(x)
    }
}
