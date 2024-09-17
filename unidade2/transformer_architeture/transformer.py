import torch.nn as nn

from unidade2.transformer_architeture.transformer_encoder import TransformerEncoder
from unidade2.transformer_architeture.transformer_decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 num_layers, d_ff, max_seq_len, dropout):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(vocab_size,
                                          d_model, num_heads, num_layers,
                                          d_ff, max_seq_len, dropout)
        self.decoder = TransformerDecoder(vocab_size,
                                          d_model, num_heads, num_layers,
                                          d_ff, max_seq_len, dropout)

    def forward(self, src, src_mask, causal_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(src, encoder_output,
                                      causal_mask, mask)
        return decoder_output