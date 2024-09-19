import torch
import torch.nn as nn
from unidade2.transformer_architeture.decoder_layer import DecoderLayer


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_feedforward):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.layer_norm(tgt)
