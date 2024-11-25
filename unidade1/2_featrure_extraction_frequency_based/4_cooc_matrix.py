import pandas as pd

words= ["gato", "gosta", "de", "peixe", "mar"]

coocurrence_data = [
    [0, 1, 1, 0, 1],  # gato
    [1, 0, 3, 1, 1],  # gosta
    [1, 3, 0, 1, 1],  # de
    [0, 1, 1, 0, 1],  # peixe
    [1, 1, 1, 1, 0]   # mar
]

cooc_matriz = pd.DataFrame(coocurrence_data, columns=words, index=words)

print("matriz de coocorrências:")
print(cooc_matriz)