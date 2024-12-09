import pandas as pd

# Frases fornecidas
sentences = [
    "gato gosta de peixe.",
    "peixe gosta de mar.",
    "mar gosta de gato."
]
def gerar_cooc_matrix(sentences):
    tokenized_sentences = [sentence.lower().replace('.', '').split() for sentence in sentences]

    cooccurrence_pairs = []

    for tokens in tokenized_sentences:
        for i in range(len(tokens) - 1):  # Pegar pares consecutivos (apenas palavras da esquerda e direita)
            cooccurrence_pairs.append((tokens[i], tokens[i + 1]))
            cooccurrence_pairs.append((tokens[i + 1], tokens[i]))  # Garantir simetria

    unique_words = sorted(set(word for tokens in tokenized_sentences for word in tokens))

    cooc_matrix = pd.DataFrame(0, index=unique_words, columns=unique_words)

    for word1, word2 in cooccurrence_pairs:
        cooc_matrix.loc[word1, word2] += 1

    return cooc_matrix


cooc_matrix = gerar_cooc_matrix(sentences)

print(cooc_matrix)

# Opcional: salvar o DataFrame em um arquivo CSV
#cooc_matrix.to_csv("coocurrence_matrix_adjacent.csv", index=True)
