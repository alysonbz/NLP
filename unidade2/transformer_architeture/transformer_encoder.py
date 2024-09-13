import torch.nn as nn
from unidade2.transformer_architeture.positional_encoding import PositionalEncoder
from unidade2.transformer_architeture.encoder_layer import EncoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, n_heads, d_ff, max_seq_len, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.positional_encoding = PositionalEncoder(d_model, max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model,n_heads,d_ff,dropout)
            for _ in range(num_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x