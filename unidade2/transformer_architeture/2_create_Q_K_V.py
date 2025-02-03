import numpy as np

# Matriz de entrada X
X = np.array([
    [1, 2, 3, 4],
    [4, 3, 2, 1],
    [1, 3, 5, 7],
    [7, 5, 3, 1]
])

# Defina o valor da dimensão do modelo (d_model) a partir da matriz X
d_model = X.shape[1]  # Número de colunas de X

# Definindo a dimensão dos pesos (geralmente igual a d_model)
d_k = d_model

# Inicializando as Matrizes de Pesos W_Q, W_K, W_V com distribuição normal (Xavier Initialization)
W_Q = np.random.randn(d_model, d_k) * np.sqrt(1 / d_model)
W_K = np.random.randn(d_model, d_k) * np.sqrt(1 / d_model)
W_V = np.random.randn(d_model, d_k) * np.sqrt(1 / d_model)

# Calcule as matrizes Q, K e V
Q = X @ W_Q  # Multiplicação da matriz X pelos pesos W_Q
K = X @ W_K  # Multiplicação da matriz X pelos pesos W_K
V = X @ W_V  # Multiplicação da matriz X pelos pesos W_V

# Exibição dos resultados
print("Matriz Q (Query):")
print(Q)
print("\nMatriz K (Key):")
print(K)
print("\nMatriz V (Value):")
print(V)
