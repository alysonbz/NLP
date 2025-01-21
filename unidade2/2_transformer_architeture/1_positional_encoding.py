import numpy as np

# Matriz de entrada X
X = np.array([
    [1, 2, 3, 4],
    [4, 3, 2, 1],
    [1, 3, 5, 7],
    [7, 5, 3, 1]
])

# Parâmetros
d_model = X.shape[1]  # Dimensão do vetor
positions = np.arange(X.shape[0])  # Índices das posições

# Inicializando PE
PE = np.zeros_like(X, dtype=float)

# TODO: Calcular os valores da matriz PE utilizando a fórmula
for pos in range(len(positions)):
    for i in range(d_model):
        if i % 2 == 0:  # Índices pares
            PE[pos, i] = ...  # Complete aqui
        else:  # Índices ímpares
            PE[pos, i] = ...  # Complete aqui

# TODO: Soma de X com PE
X_prime = ...

print("Matriz Resultante (X + PE):")
print(X_prime)
