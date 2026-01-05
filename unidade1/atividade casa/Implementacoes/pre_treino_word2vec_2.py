import gensim.downloader as api

# Carregar Glove pré-treinado (50 dimensões)
glove_vectors = api.load('glove-wiki-gigaword-50')

# Vetor de uma palavra
print("Vetor da palavra 'cat':")
print(glove_vectors['cat'])

# Similaridade
print("\nPalavras mais similares a 'cat':")
print(glove_vectors.most_similar('cat'))