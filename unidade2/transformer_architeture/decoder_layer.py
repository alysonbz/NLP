import torch.nn as nn
from unidade2.transformer_architeture.attention_mecanism import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self, model_dim, num_heads, feed_forward_dim, dropout_rate):
        super().__init__()

        self.multihead_attn = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.dropout_attn = nn.Dropout(dropout_rate)
        self.dropout_ffn = nn.Dropout(dropout_rate)

    def _apply_attention(self, x, mask):
        attn_out = self.multihead_attn(x, x, x, mask)
        attn_out = self.dropout_attn(attn_out)
        return self.norm1(x + attn_out)

    def _apply_feed_forward(self, x):
        ffn_out = self.feed_forward(x)
        ffn_out = self.dropout_ffn(ffn_out)
        return self.norm2(x + ffn_out)

    def forward(self, x, mask):
        out1 = self._apply_attention(x, mask)
        out2 = self._apply_feed_forward(out1)
        return out2
