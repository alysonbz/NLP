#from 2_positional_encoding import PositionalEncoder

import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_sequence_length):
        super(TransformerEncoder, self).__init__()
        self.embedding = nnEmbedding(vocab_size, d_model)

        self.positional_enconding = PositionalEncoder(d_model, max_sequence_length)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range (num_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_enconding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x