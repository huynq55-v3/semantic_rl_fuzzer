pub mod mlp;
pub mod transformer;

use burn::prelude::*;
use burn::tensor::Tensor;

pub enum ModelArchitecture {
    Mlp,
    Transformer,
}

#[derive(Module, Debug)]
pub enum ActorArchitecture<B: Backend> {
    Mlp(mlp::MlpActor<B>),
    Transformer(transformer::TransformerActor<B>),
}

impl<B: Backend> ActorArchitecture<B> {
    // 🌟 API ĐỔI THÀNH TENSOR 3D
    pub fn forward_with_floor(
        &self,
        sequence_state: Tensor<B, 3>,
        floor: f32,
    ) -> Vec<Tensor<B, 2>> {
        match self {
            Self::Mlp(m) => {
                let [b, s, d] = sequence_state.dims();
                // MLP ngu ngốc nên đập dẹt chiều thời gian lại: [Batch, SeqLen * StateDim]
                let flat_state = sequence_state.reshape([b, s * d]);
                m.forward_with_floor(flat_state, floor)
            }
            Self::Transformer(t) => t.forward_with_floor(sequence_state, floor),
        }
    }

    pub fn head_sizes(&self) -> Vec<usize> {
        match self {
            Self::Mlp(m) => m.heads.iter().map(|h| h.weight.dims()[1]).collect(),
            Self::Transformer(t) => t.heads.iter().map(|h| h.weight.dims()[1]).collect(),
        }
    }
}

pub const LOGIT_NOISE_MULTIPLIER: f64 = 40.0;
