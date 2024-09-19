#attention_mecanism.py
import torch
import torch.nn as nn
import math
from unidade2.transformer_architeture.first_transformer import Transformer

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model deve ser divisível por n_heads"

        self.d_head = d_model // n_heads
        self.n_heads = n_heads

        # Definindo as camadas lineares para Q, K, V e a saída
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        # Exemplo de inicialização da arquitetura Transformer (opcional)
        self.transformer = Transformer(d_model=d_model, n_heads=n_heads, num_encoder_layers=6, num_decoder_layers=6)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def compute_attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, value)

    def forward(self, query, key, value, mask=None):
        query, key, value = [self.split_heads(layer) for layer in (self.query(query), self.key(key), self.value(value))]
        output = self.compute_attention(query, key, value, mask)
        output = output.transpose(1, 2).contiguous().view(query.size(0), -1, self.d_head * self.n_heads)
        return self.out(output)
