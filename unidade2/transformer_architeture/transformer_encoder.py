#transformer_encoder.py
import torch
import torch.nn as nn
from unidade2.transformer_architeture.encoder_layer import EncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.layer_norm(src)
