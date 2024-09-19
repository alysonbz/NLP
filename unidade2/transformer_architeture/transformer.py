import torch
import torch.nn as nn
from unidade2.transformer_architeture.transformer_decoder import TransformerDecoder
from unidade2.transformer_architeture.transformer_encoder import TransformerEncoder

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, num_classes):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(d_model, n_heads, num_encoder_layers, dim_feedforward)
        self.decoder = TransformerDecoder(d_model, n_heads, num_decoder_layers, dim_feedforward)

        # A camada de classificação final
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Passa os dados pelo encoder
        memory = self.encoder(src, src_mask)

        # Passa o resultado pelo decoder
        output = self.decoder(tgt, memory, tgt_mask)

        # Aplica a camada de classificação final
        output = self.classifier(output)

        return output
