import numpy as np
import pandas as pd
from collections import Counter
import math

def preprocess_corpus(corpus):
    """Tokeniza e converte todos os termos em minúsculas"""
    tokenized_corpus = [doc.split() for doc in corpus]
    return [[word.lower() for word in doc] for doc in tokenized_corpus]

def calculate_tf(corpus, tokenized_corpus, unique_terms):
    """Calcula a matriz de Frequência de Termos (TF)"""
    tf_matrix = np.zeros((len(corpus), len(unique_terms)))
    for doc_idx, doc in enumerate(tokenized_corpus):
        term_counts = Counter(doc)
        for term, count in term_counts.items():
            if term in unique_terms:
                term_idx = unique_terms.index(term)
                tf_matrix[doc_idx, term_idx] = count / len(doc)
    return tf_matrix

def calculate_idf(corpus, tokenized_corpus, unique_terms):
    """Calcula o vetor de Frequência Inversa de Documento (IDF)"""
    doc_count = len(corpus)
    idf_vector = np.zeros(len(unique_terms))
    for idx, term in enumerate(unique_terms):
        doc_freq = sum(1 for doc in tokenized_corpus if term in doc)
        idf_vector[idx] = math.log((doc_count + 1) / (doc_freq + 1)) + 1  # Adiciona 1 para suavizar
    return idf_vector

def calculate_tf_idf(tf_matrix, idf_vector):
    """Calcula a matriz TF-IDF multiplicando as matrizes TF e IDF"""
    return tf_matrix * idf_vector

def calculate_tf_idf_table(corpus):
    # Tokeniza o corpus v
    tokenized_corpus = preprocess_corpus(corpus)
    unique_terms = sorted(set(term for doc in tokenized_corpus for term in doc))

    # Calcula TF
    tf_matrix = calculate_tf(corpus, tokenized_corpus, unique_terms)

    # Calcula IDF
    idf_vector = calculate_idf(corpus, tokenized_corpus, unique_terms)

    # Calcula a matriz TF-IDF
    tf_idf_matrix = calculate_tf_idf(tf_matrix, idf_vector)

    # Cria a tabela com a matriz TF-IDF
    tf_idf_df = pd.DataFrame(tf_idf_matrix, columns=unique_terms, index=corpus)
    return tf_idf_df

if __name__ == "__main__":
    corpus = [
        "O homem matou a pessoa do principe de Paris",
        "O principe matou o algoz de Paris"
    ]
    tf_idf_table = calculate_tf_idf_table(corpus)
    print("Matriz TF-IDF (Formato Tabela):")
    print(tf_idf_table)
