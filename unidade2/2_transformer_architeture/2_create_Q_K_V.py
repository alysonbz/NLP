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


np.random.seed(42)


# Geração das Matrizes de Pesos W_Q, W_K, W_V
# **Aqui, complete o código para gerar as matrizes de pesos usando distribuição normal ou outra técnica de inicialização**
scale = 1 / np.sqrt(d_model)


W_Q = np.random.normal(loc=0.0, scale=scale, size=(d_model, d_model))  # Inicializar a matriz de pesos W_Q com uma distribuição apropriada
W_K = np.random.normal(loc=0.0, scale=scale, size=(d_model, d_model))  # Inicializar a matriz de pesos W_K com uma distribuição apropriada
W_V = np.random.normal(loc=0.0, scale=scale, size=(d_model, d_model))  # Inicializar a matriz de pesos W_V com uma distribuição apropriada

# Calcule as matrizes Q, K e V
# **Complete o código para calcular Q, K e V multiplicando X pelas matrizes de pesos**
Q = X @ W_Q  # Substitua 'None' pela operação de multiplicação de X e W_Q
K = X @ W_K  # Substitua 'None' pela operação de multiplicação de X e W_K
V = X @ W_V  # Substitua 'None' pela operação de multiplicação de X e W_V

# Exibição dos resultados
print("W_Q:")
print(W_Q)
print("\nW_K:")
print(W_K)
print("\nW_V:")
print(W_V)

print("\nMatriz Q (Query):")
print(Q)
print("\nMatriz K (Key):")
print(K)
print("\nMatriz V (Value):")
print(V)
