import numpy as np


# Função Softmax
def softmax(x):
    # Passo 1: Subtrair o valor máximo de x para estabilidade numérica
    x_shifted = x - np.max(x, axis=-1, keepdims=True)

    # Passo 2: Calcular o exponencial de cada elemento de x
    exp_x = np.exp(x_shifted)

    # Passo 3: Normalizar os valores exponenciais
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_exp_x


def scaled_dot_product_attention(Q, K, V):
    # Passo 1: Calcular o produto escalar de Q e K^T
    matmul_QK = np.dot(Q, K.T)

    # Passo 2: Escalonar os resultados dividindo por sqrt(d_k)
    d_k = Q.shape[-1]  # Número de colunas de Q ou K
    scaled_attention_logits = matmul_QK / np.sqrt(d_k)

    # Passo 3: Aplicar softmax para obter as probabilidades
    attention_weights = softmax(scaled_attention_logits)

    # Passo 4: Multiplicar as probabilidades pela matriz de valores V
    output = np.dot(attention_weights, V)

    return output, attention_weights


# Exemplo de Matrizes Q, K e V
Q = np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 1, 1]])  # 3x3

K = np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 0, 1]])  # 3x3

V = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])  # 3x3 (ou 3x2, adapte conforme necessário)

output, attention_weights = scaled_dot_product_attention(Q, K, V)
print("Output:\n", output)
print("Attention Weights:\n", attention_weights)
