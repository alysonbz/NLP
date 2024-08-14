import numpy as np
from collections import defaultdict
import math


def count_vectorizer(corpus):
    """
    Implementa manualmente um CountVectorizer.
    Recebe um corpus (lista de documentos) e retorna o vocabulário e a matriz de contagem de palavras.
    """
    # Criar o vocabulário
    vocabulary = {}
    word_index = 0
    for document in corpus:
        for word in document.split():
            if word not in vocabulary:
                vocabulary[word] = word_index
                word_index += 1

    # Criar a matriz de contagem
    count_matrix = np.zeros((len(corpus), len(vocabulary)))

    for doc_index, document in enumerate(corpus):
        for word in document.split():
            word_idx = vocabulary[word]
            count_matrix[doc_index, word_idx] += 1

    return vocabulary, count_matrix


import numpy as np

def tfidf_vectorizer(corpus):
    """
    Implementa manualmente um TF-IDF Vectorizer.
    Recebe um corpus (lista de documentos) e retorna o vocabulário e a matriz TF-IDF.
    """
    # Passo 1: Criar o vocabulário e a matriz de contagem
    vocabulary, count_matrix = count_vectorizer(corpus)

    # Número de documentos no corpus
    num_docs = len(corpus)

    # Passo 2: Calcular o TF (Term Frequency)
    tf_matrix = np.zeros_like(count_matrix, dtype=float)  # Garanta que a matriz seja de float
    for i in range(count_matrix.shape[0]):
        total_terms_in_doc = np.sum(count_matrix[i])
        if total_terms_in_doc > 0:
            tf_matrix[i] = count_matrix[i] / total_terms_in_doc

    # Passo 3: Calcular o IDF (Inverse Document Frequency)
    df = np.sum(count_matrix > 0, axis=0)
    idf = np.log((num_docs + 1) / (df + 1)) + 1  # Suavização de IDF similar ao scikit-learn

    # Passo 4: Calcular o TF-IDF
    tfidf_matrix = tf_matrix * idf

    # Passo 5: Normalizar os vetores TF-IDF usando L2 normalization
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    tfidf_matrix = tfidf_matrix / norms

    return vocabulary, tfidf_matrix

# Exemplo de uso:
# corpus = ["este é um documento", "este documento é o segundo documento", "e este é o terceiro documento"]
# vocab, count_matrix = count_vectorizer(corpus)
# vocab, tfidf_matrix = tfidf_vectorizer(corpus)
