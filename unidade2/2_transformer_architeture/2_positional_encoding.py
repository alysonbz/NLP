#from 1_first_transformer import model
import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # matriz de codificação posicional
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))

        # funções seno e cosseno
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # dimensão para lote
        pe = pe.unsqueeze(0)

        # buffer sem gradientes
        self.register_buffer('pe', pe)

    def forward(self, x):
        # codificação posicional ao input
        x = x + self.pe[:, :x.size(1), :]
        return x
