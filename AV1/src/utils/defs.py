# Transform corpus into vectors using Word2Vec embeddings
import numpy as np


def text_to_avg_vector(text, model):
    tokens = text.split()
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)