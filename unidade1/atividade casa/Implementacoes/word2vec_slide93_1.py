from gensim.models import Word2Vec

corpus = [
    "o gato gosta de leite",
    "o cachorro gosta de osso",
    "o rato caça queijo",
    "o gato e o rato são amigos",
    "o cachorro late alto"
]

# preprocessamento: tokenizar as frases
sentences = [sentence.split() for sentence in corpus]

# Treinando o modelo Word2Vec
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

# Explorando os embeddings
print("Vetor da palavra 'gato':")
print(model.wv['gato'])

# Explorando os embeddings
print("\nVetor da palavra 'cachorro':")
print(model.wv['cachorro'])

# Similaridade entre palavras
print("\nPalavras mais similares a 'gato':")
print(model.wv.most_similar('gato'))