import numpy as np
import pandas as pd
import math

def tf(term,document):
    """Calcula a frequência do termo em um documento de forma manual."""
    words = document.split()
    termo_count = 0
    for word in words:
        if word == term:
            termo_count += 1
    total_terms = len(words)
    return termo_count / total_terms

def idf(term,corpus):
    """Calcula a frequência inversa do termo no corpus."""
    num_documents_with_term = 0
    for document in corpus:
        if term in document.split():
            num_documents_with_term +=1
    if num_documents_with_term > 0:
        return math.log(len(corpus) / num_documents_with_term)
    else:
        return 0.0

def tf_idf(corpus):
    """Calcula a matriz TF-IDF para um corpus"""
    unique_terms = list(set(term for document in corpus for term in document.split()))

    tf_idf_matrix = np.zeros((len(corpus), len(unique_terms)))

    for i, document in enumerate(corpus):
        for j, term in enumerate(unique_terms):
            tf_value = tf(term,document)
            idf_value = idf(term, corpus)
            tf_idf_matrix[i, j] = tf_value * idf_value

    tf_idf_df = pd.DataFrame(tf_idf_matrix, columns=unique_terms)
    return tf_idf_df

corpus = [
    "gato preto subiu no telhado",
    "o gato branco desceu do telhado",
    "gato preto gato branco no telhado"
]
tf_idf_matrix = tf_idf(corpus)
print(tf_idf_matrix)