import itertools
from scipy.sparse import lil_matrix

# Lista de frases
phrases = [
    "gato gosta de peixe",
    "peixe gosta de mar",
    "mar gosta de gato"
]

# Construção do vocabulário
"""vocabulary = list(set(" ".join(phrases).split()))  # Palavras únicas
main_dic = {word: idx for idx, word in enumerate(vocabulary)}  # Mapeamento de palavra para índice"""

main_dic = {'gato': 0, 'gosta': 1, 'de': 2, 'peixe': 3, 'mar': 4}


# Construção de corp_mat
corp_mat = []
for phrase in phrases:
    words = phrase.split()  # Divide a frase em palavras
    row = [(main_dic[word], 1) for word in words]  # Mapeia cada palavra para o índice
    corp_mat.append(row)

print(corp_mat)

# Passo 1: Construir o dicionário de coocorrências
trial = {}
for row in corp_mat:
    idx = [q[0] for q in row]  # Pega os índices das palavras na frase
    combos = list(itertools.combinations(idx, 2))  # Todas as combinações de pares
    for tup in combos:
        if tup in trial:
            trial[tup] += 1
        else:
            trial[tup] = 1

# Passo 2: Construir a matriz esparsa de coocorrência
scorp_mat = lil_matrix((len(main_dic), len(main_dic)))
for key, val in trial.items():
    scorp_mat[key[0], key[1]] = val

# Exibindo a matriz de coocorrência
print("Vocabulário:", main_dic)
print("\nMatriz de Coocorrência:")
print(scorp_mat.toarray())
