import numpy as np


def layer_norm(x, eps=1e-6):
    """
    Aplica Layer Normalization na última dimensão do tensor x.
    """
    # Passo 1: Média na última dimensão
    mu = np.mean(x, axis=-1, keepdims=True)

    # Passo 2: Variância na última dimensão
    sigma = np.var(x, axis=-1, keepdims=True)

    # Passo 3: Normalização
    normalized_x = (x - mu) / (np.sqrt(sigma) + eps)

    return normalized_x


def residual_connection(x, sublayer_output):
    """
    Aplica a conexão residual seguida de Layer Normalization.
    """
    # Passo 1: Conexão residual
    res = x + sublayer_output

    # Passo 2: Layer Normalization
    norm_res = layer_norm(res)

    return norm_res


# Exemplo de uso
if __name__ == "__main__":
    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

    sublayer_out = np.array([[0.5, -0.5, 1.0],
                             [1.0, 0.0, -1.0]])

    output = residual_connection(x, sublayer_out)
    print("Saída com Residual Connection e Layer Normalization:\n", output)