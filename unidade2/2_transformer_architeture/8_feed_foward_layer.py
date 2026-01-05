import numpy as np


def relu(z):
    """
    Implementa a função de ativação ReLU.
    """
    return np.maximum(0, z)


def feed_forward(x, W1, b1, W2, b2):
    """
    Implementa a camada Feed-Forward do Transformer.
    """
    # Passo 1: Primeira transformação linear
    z = np.matmul(x, W1) + b1

    # Passo 2: Ativação ReLU
    h = relu(z)

    # Passo 3: Segunda transformação linear
    output = np.matmul(h, W2) + b2

    return output


# Teste da implementação
if __name__ == "__main__":
    np.random.seed(42)

    batch_size = 2
    seq_len = 4
    d_model = 6
    d_ff = 12

    x = np.random.rand(batch_size, seq_len, d_model)

    W1 = np.random.rand(d_model, d_ff)
    b1 = np.random.rand(d_ff)
    W2 = np.random.rand(d_ff, d_model)
    b2 = np.random.rand(d_model)

    output = feed_forward(x, W1, b1, W2, b2)
    print("Saída da Feed-Forward Network:\n", output)