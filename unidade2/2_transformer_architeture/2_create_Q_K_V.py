import numpy as np

# Matriz de entrada X
X = np.array([
    [1, 2, 3, 4],
    [4, 3, 2, 1],
    [1, 3, 5, 7],
    [7, 5, 3, 1]
])

# Dimensão do modelo
d_model = X.shape[1]

# Inicialização das matrizes de pesos (distribuição normal)
np.random.seed(42)  # Para reprodutibilidade

W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)

# Cálculo de Q, K e V
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Exibição dos resultados
print("Matriz Q (Query):")
print(Q)

print("\nMatriz K (Key):")
print(K)

print("\nMatriz V (Value):")
print(V)