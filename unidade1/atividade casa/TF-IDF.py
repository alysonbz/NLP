import numpy as np
from collections import Counter
import math

def preprocess_sentences(sentences):
    """
    Pré-processa as frases, convertendo-as para minúsculas e dividindo-as em tokens.
    """
    return [sentence.lower().split() for sentence in sentences]

def compute_tf(sentence):
    """
    Calcula a frequência de termos (TF) para uma frase dada.
    """
    tf = Counter(sentence)
    total_terms = len(sentence)
    for term in tf:
        tf[term] /= total_terms
    return tf

def compute_idf(corpus):
    """
    Calcula a frequência inversa de documentos (IDF) para o corpus inteiro.
    """
    idf = {}
    num_sentences = len(corpus)
    all_terms = set(term for sentence in corpus for term in sentence)

    for term in all_terms:
        containing_sentences = sum(1 for sentence in corpus if term in sentence)
        idf[term] = math.log(num_sentences / (1 + containing_sentences))  # Adiciona 1 para evitar divisão por zero

    return idf

def compute_tf_idf(corpus):
    """
    Calcula a matriz TF-IDF para um corpus de frases dado.
    """
    # Passo 1: Calcula IDF para o corpus
    idf = compute_idf(corpus)

    # Passo 2: Calcula TF para cada frase e depois calcula TF-IDF
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
    Imprime a matriz TF-IDF de forma legível.
    """
    for i, tf_idf in enumerate(tf_idf_matrix):
        print(f"Frase {i}:")
        for term, score in tf_idf.items():
            print(f"  {term}: {score:.4f}")
        print()

if __name__ == "__main__":
    # Define um corpus sintético de frases em português
    sentences = [
        "A tecnologia evolui rapidamente ao longo dos anos",
        "A inteligência artificial está transformando diversos setores",
        "Os avanços em computação quântica são impressionantes",
        "O desenvolvimento sustentável é crucial para o futuro do planeta"
    ]

    # Pré-processa as frases: tokeniza e coloca em minúsculas
    corpus = preprocess_sentences(sentences)

    # Calcula a matriz TF-IDF e o IDF
    tf_idf_matrix, idf = compute_tf_idf(corpus)

    print("Matriz TF-IDF:")
    print_tf_idf_matrix(tf_idf_matrix)

    print("Valores IDF:")
    for term, score in idf.items():
        print(f"  {term}: {score:.4f}")
