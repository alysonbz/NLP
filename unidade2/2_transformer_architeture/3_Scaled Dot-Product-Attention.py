import numpy as np

# Função para calcular o Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V):
    # Passo 1: Produto escalar entre Q e K^T
    attention_logits = np.dot(Q, K.T)

    # Passo 2: Escalonamento por sqrt(d_k)
    d_k = Q.shape[-1]
    scaled_attention_logits = attention_logits / np.sqrt(d_k)

    # Passo 3: Softmax
    attention_weights = softmax(scaled_attention_logits)

    # Passo 4: Multiplicação pelos valores V
    output = np.dot(attention_weights, V)

    return output, attention_weights

# Função Softmax (linha a linha)
def softmax(x):
    # Estabilidade numérica
    x_shifted = x - np.max(x, axis=1, keepdims=True)

    # Exponencial
    exp_x = np.exp(x_shifted)

    # Normalização
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Exemplo de Matrizes Q, K e V
Q = np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 1, 1]])  # 3x3

K = np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 0, 1]])  # 3x3

V = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])  # 3x3

# Calculando o Scaled Dot-Product Attention
output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("Output:\n", output)
print("\nAttention Weights:\n", attention_weights)