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

# Geração das Matrizes de Pesos W_Q, W_K, W_V
# **Aqui, complete o código para gerar as matrizes de pesos usando distribuição normal ou outra técnica de inicialização**

W_Q = None  # Inicializar a matriz de pesos W_Q com uma distribuição apropriada
W_K = None  # Inicializar a matriz de pesos W_K com uma distribuição apropriada
W_V = None  # Inicializar a matriz de pesos W_V com uma distribuição apropriada

# Calcule as matrizes Q, K e V
# **Complete o código para calcular Q, K e V multiplicando X pelas matrizes de pesos**
Q = None  # Substitua 'None' pela operação de multiplicação de X e W_Q
K = None  # Substitua 'None' pela operação de multiplicação de X e W_K
V = None  # Substitua 'None' pela operação de multiplicação de X e W_V

# Exibição dos resultados
print("Matriz Q (Query):")
print(Q)
print("\nMatriz K (Key):")
print(K)
print("\nMatriz V (Value):")
print(V)
