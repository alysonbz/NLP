import numpy as np
from collections import Counter
import math


def preprocess_sentences(sentences):
    """
    Preprocess sentences by tokenizing and converting to lowercase.
    """
    return [sentence.lower().split() for sentence in sentences]


def compute_tf(sentence):
    """
    Compute term frequency (TF) for a given sentence.
    """
    tf = Counter(sentence)
    total_terms = len(sentence)
    for term in tf:
        tf[term] /= total_terms
    return tf


def compute_idf(corpus):
    """
    Compute inverse document frequency (IDF) for the entire corpus.
    """
    idf = {}
    num_sentences = len(corpus)
    all_terms = set(term for sentence in corpus for term in sentence)

    for term in all_terms:
        containing_sentences = sum(1 for sentence in corpus if term in sentence)
        idf[term] = math.log(num_sentences / (1 + containing_sentences))  # add 1 to avoid division by zero

    return idf


def compute_tf_idf(corpus):
    """
    Compute the TF-IDF matrix for a given corpus of sentences.
    """
    # Step 1: Compute IDF for the corpus
    idf = compute_idf(corpus)

    # Step 2: Compute TF for each sentence and then compute TF-IDF
    tf_idf_matrix = []

    for sentence in corpus:
        tf = compute_tf(sentence)
        tf_idf = {}
        for term, tf_value in tf.items():
            tf_idf[term] = tf_value * idf.get(term, 0)
        tf_idf_matrix.append(tf_idf)

    return tf_idf_matrix, idf


def print_tf_idf_matrix(tf_idf_matrix):
    """
    Print the TF-IDF matrix in a readable format.
    """
    for i, tf_idf in enumerate(tf_idf_matrix):
        print(f"Frase {i}:")
        for term, score in tf_idf.items():
            print(f"  {term}: {score:.4f}")
        print()


if __name__ == "__main__":
    # Define a synthetic corpus of Portuguese sentences
    sentences = [
        "O gato estava em cima do tapete",
        "O cachorro latia para o gato",
        "O gato estava feliz",
        "O cachorro perseguia o gato o dia todo"
    ]

    # Preprocess sentences: tokenize and lowercase
    corpus = preprocess_sentences(sentences)

    tf_idf_matrix, idf = compute_tf_idf(corpus)

    print("Matriz TF-IDF:")
    print_tf_idf_matrix(tf_idf_matrix)

    print("Valores IDF:")
    for term, score in idf.items():
        print(f"  {term}: {score:.4f}")
