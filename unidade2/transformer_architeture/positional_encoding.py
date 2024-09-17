import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Certifique-se de que max_seq_length e d_model são inteiros
        max_seq_length = int(max_seq_length)
        d_model = int(d_model)

        # Inicializa o tensor de codificação posicional
        pe = torch.zeros(max_seq_length, d_model)

        # Inicializa a tabela de posições (usando as fórmulas do paper original dos Transformers)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Aplicando seno nas posições pares
        pe[:, 1::2] = torch.cos(position * div_term)  # Aplicando cosseno nas posições ímpares

        pe = pe.unsqueeze(0)  # Adiciona dimensão para o batch
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adiciona a codificação posicional aos embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)