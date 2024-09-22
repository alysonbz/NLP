import torch
import torch.nn as nn
from unidade2.transformer_architeture.encoder_layer import EncoderLayer
from unidade2.transformer_architeture.positional_encoding import PositionalEncoder

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers, num_heads, feed_forward_dim, dropout_rate, max_sequence_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = PositionalEncoder(model_dim, max_sequence_length)
        self.layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, feed_forward_dim, dropout_rate) for _ in range(num_layers)]
        )

    def _embed_and_position(self, x):
        x = self.embedding(x)
        return self.positional_encoding(x)

    def _apply_layers(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

    def forward(self, x, mask):
        x = self._embed_and_position(x)
        x = self._apply_layers(x, mask)
        return x
