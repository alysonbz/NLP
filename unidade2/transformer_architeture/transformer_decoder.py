import torch.nn as nn
import torch.nn.functional as F
from unidade2.transformer_architeture.positional_encoding import PositionalEncoder
from unidade2.transformer_architeture.decoder_layer import DecoderLayer


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_sequence_length):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoder(d_model, max_sequence_length)

        # Verifica o valor antes de passá-lo para DecoderLayer
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, self_mask, encoder_output=None, cross_attention_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        # Loop pelas camadas do decodificador
        for layer in self.layers:
            # Passe o encoder_output e cross_attention_mask se forem necessários nas camadas do decoder
            x = layer(x, self_mask)  # Ajuste conforme necessário para o uso desses parâmetros

        x = self.fc(x)
        return F.log_softmax(x, dim=-1)