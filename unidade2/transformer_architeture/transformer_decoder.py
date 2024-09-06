import torch.nn as nn
from transformers.models.transfo_xl.modeling_transfo_xl import PositionalEmbedding
import torch.nn.functional as F

from unidade2.transformer_architeture.decoder_layer import DecoderLayer


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model,num_layers, num_heads,d_ff,dropout,max_sequence_length):
        super(TransformerDecoder, self), self.__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEmbedding(d_model, max_sequence_length)
        self.layers = nn.ModuleList([DecoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, self_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, self_mask)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)