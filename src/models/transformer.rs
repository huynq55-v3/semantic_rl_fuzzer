use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu,
};
use burn::prelude::*;

use crate::models::LOGIT_NOISE_MULTIPLIER;

#[derive(Module, Debug)]
pub struct TransformerActor<B: Backend> {
    pub embedding: Linear<B>,
    pub pos_embedding: Embedding<B>,
    pub attention: MultiHeadAttention<B>,
    pub layer_norm: LayerNorm<B>,
    pub heads: Vec<Linear<B>>,
    pub relu: Relu,
}

impl<B: Backend> TransformerActor<B> {
    pub fn new(
        device: &B::Device,
        input_size: usize,
        d_model: usize,
        head_sizes: &[usize],
        max_seq_len: usize,
    ) -> Self {
        let embedding = LinearConfig::new(input_size, d_model).init(device);
        let pos_embedding = EmbeddingConfig::new(max_seq_len, d_model).init(device);
        let attention = MultiHeadAttentionConfig::new(d_model, 4).init(device);
        let layer_norm = LayerNormConfig::new(d_model).init(device);

        let mut heads = Vec::new();
        for &size in head_sizes {
            heads.push(LinearConfig::new(d_model, size).init(device));
        }

        Self {
            embedding,
            pos_embedding,
            attention,
            layer_norm,
            heads,
            relu: Relu::new(),
        }
    }

    // 🌟 CHUYỂN NHẬN TENSOR 3D: [Batch, SequenceLength, StateDim]
    pub fn forward_with_floor(
        &self,
        sequence_state: Tensor<B, 3>,
        floor: f32,
    ) -> Vec<Tensor<B, 2>> {
        let [batch, seq_len, _] = sequence_state.dims();

        // 1. Nhúng State và Vị trí thời gian
        let x = self.embedding.forward(sequence_state);
        let positions = Tensor::arange(0..seq_len as i64, &x.device())
            .unsqueeze_dim(0)
            .repeat(&[batch, 1]);
        let pos_embeds = self.pos_embedding.forward(positions);

        let x = self.relu.forward(x + pos_embeds);

        // 2. Chú ý (Self-Attention) nhìn lại quá khứ
        let mha_input = MhaInput::self_attn(x.clone());
        let attn_out = self.attention.forward(mha_input).context;

        let x = self.layer_norm.forward(x + attn_out);

        // 3. 🌟 GOM KẾT QUẢ TỪ BƯỚC THỜI GIAN CUỐI CÙNG (HIỆN TẠI)
        // Lấy kích thước ra trước để tránh lỗi "borrow of moved value"
        let dims = x.dims();
        let d_model_inner = dims[2];

        let last_token_feature = x
            .slice([0..batch, seq_len - 1..seq_len, 0..d_model_inner])
            .reshape([batch, d_model_inner]);

        self.heads
            .iter()
            .map(|head| {
                let logits = head.forward(last_token_feature.clone());
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
