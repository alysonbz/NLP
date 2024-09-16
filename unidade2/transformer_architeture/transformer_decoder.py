import numpy as np


class TransformerDecoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, dropout_rate=0.1):
        """
        Inicializa a classe TransformerDecoder.

        :param num_layers: Número de camadas de decodificador.
        :param d_model: Dimensão do modelo (tamanho dos vetores de embedding).
        :param num_heads: Número de cabeças na atenção multi-cabeça.
        :param d_ff: Dimensão da camada feedforward.
        :param vocab_size: Tamanho do vocabulário para a camada de saída.
        :param dropout_rate: Taxa de dropout para regularização.
        """
        # Inicializa a lista de camadas de decodificador
        self.layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

        # Inicializa a camada final de projeção (Linear)
        self.final_layer = np.random.randn(d_model, vocab_size) * 0.01

    def forward(self, x, enc_output, mask=None):
        """
        Realiza a propagação para frente através da camada TransformerDecoder.

        :param x: Entrada do decodificador (matriz numpy com formato (batch_size, seq_len, d_model)).
        :param enc_output: Saída do codificador (matriz numpy com formato (batch_size, seq_len, d_model)).
        :param mask: Máscara de atenção opcional.
        :return: Saída final do decodificador (matriz numpy com formato (batch_size, seq_len, vocab_size)).
        """
        # Propaga a entrada através das camadas do decodificador
        for layer in self.layers:
            x = layer.forward(x, enc_output, mask)

        # Aplica a camada final de projeção para gerar a saída final
        logits = np.dot(x, self.final_layer)

        return logits
