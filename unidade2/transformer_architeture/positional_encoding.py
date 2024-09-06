import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Criação da matriz de posições
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # Cálculo do termo de divisão
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Aplicação de seno e cosseno nas posições
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Expansão das dimensões para facilitar a adição ao embedding
        pe = pe.unsqueeze(0)

        # Registrar pe como buffer (não será atualizado durante o treinamento)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adicionar o encoding posicional ao embedding
        x = x + self.pe[:, :x.size(1)]
        return x