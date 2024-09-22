import torch.nn as nn
from unidade2.transformer_architeture.transformer_encoder import TransformerEncoder
from unidade2.transformer_architeture.transformer_decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers, feed_forward_dim, max_seq_len, dropout_rate):
        super().__init__()

        self.encoder = self._initialize_encoder(vocab_size, model_dim, num_heads, num_layers, feed_forward_dim, max_seq_len, dropout_rate)
        self.decoder = self._initialize_decoder(vocab_size, model_dim, num_heads, num_layers, feed_forward_dim, max_seq_len, dropout_rate)

    def _initialize_encoder(self, vocab_size, model_dim, num_heads, num_layers, feed_forward_dim, max_seq_len, dropout_rate):
        return TransformerEncoder(vocab_size, model_dim, num_heads, num_layers, feed_forward_dim, max_seq_len, dropout_rate)

    def _initialize_decoder(self, vocab_size, model_dim, num_heads, num_layers, feed_forward_dim, max_seq_len, dropout_rate):
        return TransformerDecoder(vocab_size, model_dim, num_heads, num_layers, feed_forward_dim, max_seq_len, dropout_rate)

    def forward(self, src, src_mask, causal_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(src, encoder_output, causal_mask)
        return decoder_output
