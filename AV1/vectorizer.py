import numpy as np
from collections import Counter


def build_vocabulary(texts):
    """
    Cria um vocabulário a partir de uma lista de textos.
    """
    vocab = set()
    for text in texts:
        vocab.update(text.split())
    return sorted(vocab)


def count_vectorizer(texts):
    """
    Transforma uma lista de textos em uma matriz de contagem de palavras.
    """
    vocab = build_vocabulary(texts)
    vocab_index = {word: idx for idx, word in enumerate(vocab)}

    # Inicializar a matriz de contagem
    matrix = np.zeros((len(texts), len(vocab)))

    for i, text in enumerate(texts):
        word_counts = Counter(text.split())
        for word, count in word_counts.items():
            if word in vocab_index:
                matrix[i, vocab_index[word]] = count

    return matrix, vocab


# Exemplo de uso
texts = ["o cachorro correu", "o gato correu rápido", "o cachorro e o gato correram"]
count_matrix, vocab = count_vectorizer(texts)
print("Matriz de Contagem:\n", count_matrix)
print("Vocabulário:\n", vocab)

import numpy as np
from collections import Counter
from math import log


def build_vocabulary(texts):
    """
    Cria um vocabulário a partir de uma lista de textos.
    """
    vocab = set()
    for text in texts:
        vocab.update(text.split())
    return sorted(vocab)


def compute_tf(text, vocab):
    """
    Calcula a frequência de termos (TF) para um texto.
    """
    tf = Counter(text.split())
    total_terms = len(text.split())
    tf_vector = np.zeros(len(vocab))

    for word, count in tf.items():
        if word in vocab:
            tf_vector[vocab[word]] = count / total_terms

    return tf_vector


def compute_idf(texts, vocab):
    """
    Calcula a frequência inversa de documentos (IDF) para o vocabulário.
    """
    idf = np.zeros(len(vocab))
    num_docs = len(texts)

    for i, word in enumerate(vocab):
        doc_count = sum(word in text.split() for text in texts)
        idf[i] = log(num_docs / (1 + doc_count))

    return idf


def tfidf_vectorizer(texts):
    """
    Transforma uma lista de textos em uma matriz TF-IDF.
    """
    vocab_list = build_vocabulary(texts)
    vocab_index = {word: idx for idx, word in enumerate(vocab_list)}

    tfidf_matrix = np.zeros((len(texts), len(vocab_list)))

    idf = compute_idf(texts, vocab_index)

    for i, text in enumerate(texts):
        tf = compute_tf(text, vocab_index)
        tfidf_matrix[i] = tf * idf

    return tfidf_matrix, vocab_list


# Exemplo de uso
texts = ["o cachorro correu", "o gato correu rápido", "o cachorro e o gato correram"]
tfidf_matrix, vocab = tfidf_vectorizer(texts)
print("Matriz TF-IDF:\n", tfidf_matrix)
print("Vocabulário:\n", vocab)
