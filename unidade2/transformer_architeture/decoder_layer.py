import numpy as np


class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        """
        Inicializa a camada DecoderLayer.

        :param d_model: Dimensão do modelo (tamanho dos vetores de embedding).
        :param num_heads: Número de cabeças na atenção multi-cabeça.
        :param d_ff: Dimensão da camada feedforward.
        :param dropout_rate: Taxa de dropout para regularização.
        """
        # Inicializando camadas de atenção e feedforward
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardSubLayer(d_model, d_ff)

        # Inicializando camadas de normalização
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

        # Inicializando dropout
        self.dropout = Dropout(dropout_rate)

    def forward(self, x, enc_output, mask=None):
        """
        Realiza a propagação para frente através da camada DecoderLayer.

        :param x: Entrada do decodificador (matriz numpy com formato (batch_size, seq_len, d_model)).
        :param enc_output: Saída do codificador (matriz numpy com formato (batch_size, seq_len, d_model)).
        :param mask: Máscara de atenção opcional.
        :return: Saída da camada DecoderLayer.
        """
        # Atenção auto-regressiva
        attn_output = self.self_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)

        # Atenção cruzada
        cross_attn_output = self.cross_attention(x, enc_output, enc_output, mask)
        cross_attn_output = self.dropout(cross_attn_output)
        x = self.norm2(x + cross_attn_output)

        # Passagem pela camada feedforward
        ff_output = self.feed_forward.forward(x)
        x = self.dropout(ff_output)

        return x
