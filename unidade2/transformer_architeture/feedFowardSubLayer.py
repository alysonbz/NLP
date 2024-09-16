import numpy as np


class FeedForwardSubLayer:
    def __init__(self, input_dim, output_dim):
        """
        Inicializa a camada FeedForwardSubLayer.

        :param input_dim: Número de unidades na camada de entrada.
        :param output_dim: Número de unidades na camada de saída.
        """
        # Inicializando pesos e viés com valores aleatórios pequenos
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))

    def forward(self, X):
        """
        Realiza a propagação para frente através da camada.

        :param X: Dados de entrada (matriz numpy com formato (batch_size, input_dim)).
        :return: Saída da camada após a aplicação da função linear (matriz numpy com formato (batch_size, output_dim)).
        """
        # Calcula a saída da camada
        self.input = X  # Armazena a entrada para possível uso posterior
        self.output = np.dot(X, self.weights) + self.biases
        return self.output


