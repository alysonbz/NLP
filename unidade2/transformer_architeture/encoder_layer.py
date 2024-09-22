import torch
import torch.nn as nn
from unidade2.transformer_architeture.attention_mecanism import MultiHeadAttention
from unidade2.transformer_architeture.feedFowardSubLayer import FeedForwardSublayer

class Encoder(nn.Module):
    def __init__(self, model_dim, num_heads, feed_forward_dim, dropout_rate):
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = FeedForwardSublayer(model_dim, feed_forward_dim)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def _apply_self_attention(self, x, mask):
        attn_output = self.self_attention(x, x, x, mask)
        return self.layer_norm1(x + self.dropout(attn_output))

    def _apply_feed_forward(self, x):
        ff_output = self.feed_forward(x)
        return self.layer_norm2(x + self.dropout(ff_output))

    def forward(self, x, mask):
        x = self._apply_self_attention(x, mask)
        x = self._apply_feed_forward(x)
        return x
