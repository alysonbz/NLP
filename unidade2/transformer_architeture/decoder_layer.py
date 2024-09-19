# decoder_layer.py
import torch
import torch.nn as nn
from unidade2.transformer_architeture.attention_mecanism import MultiHeadAttention  # Importação absoluta
from unidade2.transformer_architeture.feedFowardSubLayer import FeedForwardSubLayer

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForwardSubLayer(d_model, dim_feedforward)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        self_attn_output = self.self_attention(tgt, tgt, tgt)
        tgt = self.layer_norm1(tgt + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attention(tgt, memory, memory)
        tgt = self.layer_norm2(tgt + self.dropout(cross_attn_output))
        return self.feed_forward(self.layer_norm3(tgt))
