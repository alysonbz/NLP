import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, model_dim, max_seq_length, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Garante que max_seq_length e model_dim são inteiros
        max_seq_length = int(max_seq_length)
        model_dim = int(model_dim)

        self.pe = self._create_positional_encoding(max_seq_length, model_dim)

    def _create_positional_encoding(self, max_seq_length, model_dim):
        pe = torch.zeros(max_seq_length, model_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))

        pe[:, 0::2] = torch.sin(position * div_term)  # Seno para posições pares
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosseno para posições ímpares

        return pe.unsqueeze(0)  # Adiciona dimensão para o batch

    def forward(self, x):
        # Adiciona a codificação posicional aos embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
