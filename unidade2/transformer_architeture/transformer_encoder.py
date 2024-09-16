import torch
import torch.nn as nn
from encoder_layer import EncoderLayer  # Certifique-se que esse caminho esteja correto


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # Lista de camadas EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Normalização final
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Tensor de entrada de forma (batch_size, seq_len, d_model).
            src_mask: Máscara opcional para a atenção.
        """
        for layer in self.layers:
            src = layer(src, src_mask)

        # Aplicar a normalização final após todas as camadas
        output = self.norm(src)

        return output
