# encoder_transformer.py
import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class Encoder(nn.Module):
    def __init__(self, d_model=512, n_heads=2, num_layers=6, max_len=5000):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoder(d_model, max_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads),
            num_layers=num_layers
        )

    def forward(self, src, src_key_padding_mask=None):
        src = self.pos_encoder(src)
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)



