#encoder_layer.py
import torch
import torch.nn as nn
from unidade2.transformer_architeture.attention_mecanism import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, src_mask=None):
        attn_output = self.self_attention(src, src, src)
        src = self.layer_norm1(src + self.dropout(attn_output))
        ff_output = self.feed_forward(src)
        return self.layer_norm2(src + self.dropout(ff_output))
