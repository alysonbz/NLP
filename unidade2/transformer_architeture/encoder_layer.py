import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_mecanism import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # Atenção Multi-cabeça
        self.self_attention = MultiHeadAttention(d_model, n_heads)

        # Normalização
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Rede feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Tensor de entrada de forma (batch_size, seq_len, d_model).
            src_mask: Máscara opcional para a atenção.
        """
        # Atenção com máscara, se fornecida
        attn_output, _ = self.self_attention(src, src, src, src_mask)
        attn_output = self.dropout(attn_output)
        src = self.norm1(src + attn_output)  # Residual connection + LayerNorm

        # Rede feedforward
        ff_output = self.feedforward(src)
        ff_output = self.dropout(ff_output)
        src = self.norm2(src + ff_output)  # Residual connection + LayerNorm

        return src
