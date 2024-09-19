# encoder_decoder.py
import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class Encoder(nn.Module):
    def __init__(self, d_model=512, n_heads=2, num_layers=6, max_len=5000):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoder(d_model, max_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads),
            num_layers=num_layers
        )

    def forward(self, src, src_key_padding_mask=None):
        src = self.pos_encoder(src)
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

class Decoder(nn.Module):
    def __init__(self, d_model=512, n_heads=2, num_layers=6, max_len=5000):
        super(Decoder, self).__init__()
        self.pos_encoder = PositionalEncoder(d_model, max_len)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads),
            num_layers=num_layers
        )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.pos_encoder(tgt)
        return self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

class EncoderDecoder(nn.Module):
    def __init__(self, d_model=512, n_heads=2, num_layers=6, max_len=5000):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(d_model, n_heads, num_layers, max_len)
        self.decoder = Decoder(d_model, n_heads, num_layers, max_len)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output


