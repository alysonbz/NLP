import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def _reshape_heads(self, x):
        return x.view(x.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

    def _calculate_attention(self, q, k, mask):
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            attn_scores.masked_fill_(mask == 0, float("-inf"))
        return F.softmax(attn_scores, dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self._reshape_heads(self.q_linear(query)).reshape(batch_size * self.num_heads, -1, self.head_dim)
        k = self._reshape_heads(self.k_linear(key)).reshape(batch_size * self.num_heads, -1, self.head_dim)
        v = self._reshape_heads(self.v_linear(value)).reshape(batch_size * self.num_heads, -1, self.head_dim)

        attn_weights = self._calculate_attention(q, k, mask)

        out = torch.matmul(attn_weights, v).view(batch_size, self.num_heads, -1, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
        return self.out_linear(out)