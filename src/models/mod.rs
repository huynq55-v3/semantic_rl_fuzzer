pub mod mlp;

use burn::prelude::*;
use burn::tensor::Tensor;

pub enum ModelArchitecture {
    Mlp,
}

#[derive(Module, Debug)]
pub enum ActorArchitecture<B: Backend> {
    Mlp(mlp::MlpActor<B>),
}

impl<B: Backend> ActorArchitecture<B> {
    pub fn forward_with_floor(&self, state: Tensor<B, 2>, floor: f32) -> Vec<Tensor<B, 2>> {
        match self {
            Self::Mlp(m) => m.forward_with_floor(state, floor),
        }
    }

    pub fn head_sizes(&self) -> Vec<usize> {
        match self {
            Self::Mlp(m) => m.heads.iter().map(|h| h.weight.dims()[1]).collect(),
        }
    }
}

pub const LOGIT_NOISE_MULTIPLIER: f64 = 40.0;
