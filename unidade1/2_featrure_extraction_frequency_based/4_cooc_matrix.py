import itertools  # Adicione essa linha para importar itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import lil_matrix

# Lista de frases
phrases = [
    "gato gosta de peixe",
    "peixe gosta de mar",
    "mar gosta de gato"
]

# Construção do vocabulário (utilizando set para garantir palavras únicas)
vocabulary = sorted(set(" ".join(phrases).split()))  # Palavras únicas
main_dic = {word: idx for idx, word in enumerate(vocabulary)}  # Mapeamento de palavra para índice

# Construção de corp_mat usando list comprehension
corp_mat = [
    [(main_dic[word], 1) for word in phrase.split()]
    for phrase in phrases
]

# Passo 1: Construir o dicionário de coocorrências utilizando defaultdict
cooccurrence_dict = defaultdict(int)
for row in corp_mat:
    idx = [word[0] for word in row]  # Pega os índices das palavras na frase
    for i, j in itertools.combinations(idx, 2):  # Combinações de dois índices
        cooccurrence_dict[tuple(sorted((i, j)))] += 1  # Ordena os índices para garantir simetria

# Passo 2: Construir a matriz esparsa de coocorrência
scorp_mat = lil_matrix((len(main_dic), len(main_dic)))
for (i, j), count in cooccurrence_dict.items():
    scorp_mat[i, j] = count
    scorp_mat[j, i] = count  # Matriz simétrica

# Exibindo resultados
print("Vocabulário:", main_dic)
print("\nMatriz de Coocorrência:")
print(scorp_mat.toarray())

# Exibindo a matriz como um DataFrame para melhor visualização
df = pd.DataFrame(scorp_mat.toarray(), index=vocabulary, columns=vocabulary)
print("\nMatriz de Coocorrência (DataFrame):")
print(df)
