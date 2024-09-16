import numpy as np
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer
from attention_mecanism import MultiHeadAttention
from feedforward_sub_layer import FeedForwardSubLayer
from layer_normalization import LayerNormalization
from dropout import Dropout


class TransformerEncoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout_rate=0.1):
        """
        Inicializa a classe TransformerEncoder.

        :param num_layers: Número de camadas do codificador.
        :param d_model: Dimensão do modelo (tamanho dos vetores de embedding).
        :param num_heads: Número de cabeças na atenção multi-cabeça.
        :param d_ff: Dimensão da camada feedforward.
        :param dropout_rate: Taxa de dropout para regularização.
        """
        self.layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask=None):
        """
        Realiza a propagação para frente através da camada TransformerEncoder.

        :param x: Entrada para o codificador (matriz numpy com formato (batch_size, seq_len, d_model)).
        :param mask: Máscara opcional para a entrada.
        :return: Saída do codificador.
        """
        for layer in self.layers:
            x = layer.forward(x, mask)
        return self.norm(x)


class TransformerDecoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, dropout_rate=0.1):
        """
        Inicializa a classe TransformerDecoder.

        :param num_layers: Número de camadas do decodificador.
        :param d_model: Dimensão do modelo (tamanho dos vetores de embedding).
        :param num_heads: Número de cabeças na atenção multi-cabeça.
        :param d_ff: Dimensão da camada feedforward.
        :param vocab_size: Tamanho do vocabulário para a camada de saída.
        :param dropout_rate: Taxa de dropout para regularização.
        """
        self.layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        self.final_layer = np.random.randn(d_model, vocab_size) * 0.01
        self.norm = LayerNormalization(d_model)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x, enc_output, mask=None):
        """
        Realiza a propagação para frente através da camada TransformerDecoder.

        :param x: Entrada para o decodificador (matriz numpy com formato (batch_size, seq_len, d_model)).
        :param enc_output: Saída do codificador (matriz numpy com formato (batch_size, seq_len, d_model)).
        :param mask: Máscara opcional para a entrada do decodificador.
        :return: Saída final do decodificador.
        """
        for layer in self.layers:
            x = layer.forward(x, enc_output, mask)
        logits = np.dot(self.dropout(self.norm(x)), self.final_layer)
        return logits


class Transformer:
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, d_ff, vocab_size, dropout_rate=0.1):
        """
        Inicializa a classe Transformer.

        :param num_encoder_layers: Número de camadas do codificador.
        :param num_decoder_layers: Número de camadas do decodificador.
        :param d_model: Dimensão do modelo (tamanho dos vetores de embedding).
        :param num_heads: Número de cabeças na atenção multi-cabeça.
        :param d_ff: Dimensão da camada feedforward.
        :param vocab_size: Tamanho do vocabulário para a camada de saída.
        :param dropout_rate: Taxa de dropout para regularização.
        """
        self.encoder = TransformerEncoder(num_layers=num_encoder_layers, d_model=d_model, num_heads=num_heads,
                                          d_ff=d_ff, dropout_rate=dropout_rate)
        self.decoder = TransformerDecoder(num_layers=num_decoder_layers, d_model=d_model, num_heads=num_heads,
                                          d_ff=d_ff, vocab_size=vocab_size, dropout_rate=dropout_rate)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Realiza a propagação para frente através da camada Transformer.

        :param src: Entrada para o codificador (matriz numpy com formato (batch_size, src_seq_len, d_model)).
        :param tgt: Entrada para o decodificador (matriz numpy com formato (batch_size, tgt_seq_len, d_model)).
        :param src_mask: Máscara opcional para a entrada do codificador.
        :param tgt_mask: Máscara opcional para a entrada do decodificador.
        :return: Saída final do Transformer (matriz numpy com formato (batch_size, tgt_seq_len, vocab_size)).
        """
        enc_output = self.encoder.forward(src, src_mask)
        logits = self.decoder.forward(tgt, enc_output, tgt_mask)
        return logits
