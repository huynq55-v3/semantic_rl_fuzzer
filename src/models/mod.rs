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
    pub fn forward_with_floor(&self, state: Tensor<B, 2>, floor: f32) -> Vec<Tensor<B, 2>> {
        match self {
            Self::Mlp(m) => m.forward_with_floor(state, floor),
            Self::Transformer(m) => m.forward_with_floor(state, floor),
        }
    }

    pub fn head_sizes(&self) -> Vec<usize> {
        match self {
            Self::Mlp(m) => m.heads.iter().map(|h| h.weight.dims()[1]).collect(),
            Self::Transformer(m) => m.heads.iter().map(|h| h.weight.dims()[1]).collect(),
        }
    }
}

#[derive(Module, Debug)]
pub enum ForwardArchitecture<B: Backend> {
    Mlp(mlp::MlpForward<B>),
    Transformer(transformer::TransformerForward<B>),
}

impl<B: Backend> ForwardArchitecture<B> {
    pub fn forward_step(&self, state: Tensor<B, 2>, action_indices: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Self::Mlp(m) => m.forward(state, action_indices),
            Self::Transformer(m) => m.forward(state, action_indices),
        }
    }

    pub fn reset_architecture(
        &self,
        device: &B::Device,
        input_size: usize,
        total_action_dims: usize,
        d_model: usize,
    ) -> Self {
        match self {
            Self::Mlp(_) => Self::Mlp(mlp::MlpForward::new(
                device,
                input_size,
                total_action_dims,
                d_model,
            )),
            Self::Transformer(_) => Self::Transformer(transformer::TransformerForward::new(
                device,
                input_size,
                total_action_dims,
                d_model,
            )),
        }
    }
}
