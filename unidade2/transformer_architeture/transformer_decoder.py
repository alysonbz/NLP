import torch.nn as nn
import torch.nn.functional as F
from unidade2.transformer_architeture.positional_encoding import PositionalEncoder
from unidade2.transformer_architeture.decoder_layer import DecoderLayer

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers, num_heads, feed_forward_dim, dropout_rate, max_sequence_length):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = PositionalEncoder(model_dim, max_sequence_length)
        self.layers = nn.ModuleList([DecoderLayer(model_dim, num_heads, feed_forward_dim, dropout_rate) for _ in range(num_layers)])
        self.fc = nn.Linear(model_dim, vocab_size)

    def _embed_and_position(self, x):
        x = self.embedding(x)
        return self.positional_encoding(x)

    def _apply_layers(self, x, self_mask):
        for layer in self.layers:
            x = layer(x, self_mask)
        return x

    def forward(self, x, self_mask, encoder_output=None, cross_attention_mask=None):
        x = self._embed_and_position(x)
        x = self._apply_layers(x, self_mask)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
