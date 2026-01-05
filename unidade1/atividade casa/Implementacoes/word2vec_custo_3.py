from gensim.models import Word2Vec

corpus = [
    "o gato gosta de leite",
    "o cachorro gosta de osso",
    "o rato caça queijo",
    "o gato e o rato são amigos",
    "o cachorro late alto"
]

# Tokenizar
sentences = [sentence.split() for sentence in corpus]

model = Word2Vec(sentences, vector_size=100, window=3, min_count=1, sg=1)
print("Vetor da palavra 'gato':", model.wv['gato'])