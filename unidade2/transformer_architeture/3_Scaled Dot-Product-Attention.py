import numpy as np


# Função para calcular o Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V):
    # Passo 1: Calcular o produto escalar de Q e K^T
    # **Complete o cálculo aqui**
    attention_logits = np.dot(Q, K.T)  # Produto escalar entre Q e K^T

    # Passo 2: Escalonar os resultados dividindo por sqrt(d_k)
    d_k = Q.shape[-1]  # Número de colunas de Q ou K
    scaled_attention_logits = attention_logits / np.sqrt(d_k)  # **Complete esta parte**

    # Passo 3: Aplicar softmax para obter as probabilidades
    attention_weights = softmax(scaled_attention_logits)  # **Função softmax incompleta**

    # Passo 4: Multiplicar as probabilidades pela matriz de valores V
    output = np.dot(attention_weights, V)

    return output, attention_weights


# Função Softmax (incompleta)
def softmax(x):
    # Passo 1: Subtrair o valor máximo de x para estabilidade numérica
    # **Complete este passo**
    x -= np.max(x, axis=1, keepdims=True)

    # Passo 2: Calcular o exponencial de cada elemento de x
    # **Complete este passo**
    exp = np.exp(x)

    # Passo 3: Normalizar os valores exponenciais, dividindo cada valor pelo somatório dos exponenciais
    # **Complete este passo**
    sum_exp = np.sum(exp, axis=-1, keepdims=True)
    softmax_result = exp / sum_exp

    return softmax_result  # Retorne a versão normalizada de x


# Exemplo de Matrizes Q, K e V
Q = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])  # 3x3
K = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])  # 3x3
V = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3x2

# Calculando o Scaled Dot-Product Attention
output, attention_weights = scaled_dot_product_attention(Q, K, V)
print("Output:\n", output)
print("Attention Weights:\n", attention_weights)
