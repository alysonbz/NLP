import numpy as np
from collections import Counter
import re
from typing import List, Tuple
from datasets import load_dataset
#

def build_vocabulary(documents: List[str]) -> List[str]:
    """
    Constrói um vocabulário a partir de uma lista de documentos.

    Args:
        documents (List[str]): Lista de documentos (textos).

    Returns:
        List[str]: Lista de palavras únicas no vocabulário, ordenada alfabeticamente.
    """
    word_freq = Counter()
    for doc in documents:
        if doc:  # Verifica se o documento não está vazio
            words = re.findall(r'\b\w+\b', doc.lower())  # Tokeniza e converte para minúsculas
            word_freq.update(words)
    return sorted(word_freq.keys())


def count_vectorizer(documents: List[str], vocab: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Converte uma lista de documentos em uma matriz de contagem de palavras.

    Args:
        documents (List[str]): Lista de documentos (textos).
        vocab (List[str], opcional): Vocabulário a ser usado para a vetorização. Se None, o vocabulário é construído a partir dos documentos.

    Returns:
        Tuple[np.ndarray, List[str]]: Tupla contendo a matriz de contagem e o vocabulário.
    """
    if vocab is None:
        vocabulary = build_vocabulary(documents)
    else:
        vocabulary = vocab

    vocab_size = len(vocabulary)
    vocab_index = {word: i for i, word in enumerate(vocabulary)}

    matrix = np.zeros((len(documents), vocab_size), dtype=int)

    for i, doc in enumerate(documents):
        words = re.findall(r'\b\w+\b', doc.lower())
        word_counts = Counter(words)
        for word, count in word_counts.items():
            if word in vocab_index:
                matrix[i, vocab_index[word]] = count

    return matrix, vocabulary


def tfidf_vectorizer(documents: List[str], vocab: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Converte uma lista de documentos em uma matriz TF-IDF.

    Args:
        documents (List[str]): Lista de documentos (textos).
        vocab (List[str], opcional): Vocabulário a ser usado para a vetorização. Se None, o vocabulário é construído a partir dos documentos.

    Returns:
        Tuple[np.ndarray, List[str]]: Tupla contendo a matriz TF-IDF e o vocabulário.
    """
    if vocab is None:
        matrix, vocabulary = count_vectorizer(documents)
    else:
        # Construa a matriz de contagem usando o vocabulário fornecido
        matrix, vocabulary = count_vectorizer(documents, vocab=vocab)

    vocab_size = len(vocabulary)
    num_docs = len(documents)

    # Calcula TF
    tf_matrix = matrix.astype(float)

    # Evita divisão por zero em documentos vazios
    row_sums = tf_matrix.sum(axis=1, keepdims=True)
    np.divide(tf_matrix, row_sums, out=tf_matrix, where=row_sums != 0)

    # Calcula IDF
    doc_freq = np.sum(matrix > 0, axis=0)
    idf = np.log((num_docs + 1) / (doc_freq + 1)) + 1  # +1 para evitar divisão por zero

    # Calcula TF-IDF
    tfidf_matrix = tf_matrix * idf

    return tfidf_matrix, vocabulary


# Carregar o dataset
ds = load_dataset("johnidouglas/twitter-sentiment-pt-BR-md-2-l")
df = ds['train'].to_pandas()

# Extrair as primeiras linhas do dataset
num_samples = 10  # Defina o número de linhas desejadas
texts = df['tweet_text'].head(num_samples).tolist()

# Aplicar Count Vectorizer
count_matrix, vocab = count_vectorizer(texts)
print("Matriz de Contagem:\n", count_matrix)
print("Vocabulário:\n", vocab)

# Aplicar TF-IDF Vectorizer
tfidf_matrix, vocab = tfidf_vectorizer(texts)
print("Matriz TF-IDF:\n", tfidf_matrix)
print("Vocabulário:\n", vocab)

import pandas as pd
import numpy as np

# Definir o número de amostras e palavras a serem visualizadas
num_samples = 5  # Número de amostras (documentos)
num_features = 20  # Número de palavras (características) para visualizar

# Gerar índices aleatórios para amostras e vocabulário
random_sample_indices = np.random.choice(count_matrix.shape[0], num_samples, replace=False)
random_feature_indices = np.random.choice(count_matrix.shape[1], num_features, replace=False)

# Selecionar uma amostra aleatória das matrizes
sample_count_matrix = count_matrix[random_sample_indices, :][:, random_feature_indices]
sample_tfidf_matrix = tfidf_matrix[random_sample_indices, :][:, random_feature_indices]

# Selecionar um subconjunto aleatório do vocabulário para visualização
sample_vocab = [vocab[i] for i in random_feature_indices]

# Criar DataFrames para visualização
df_count_matrix = pd.DataFrame(sample_count_matrix, columns=sample_vocab)
df_tfidf_matrix = pd.DataFrame(sample_tfidf_matrix, columns=sample_vocab)

# Salvar como HTML
df_count_matrix.to_html("sample_count_matrix.html")
df_tfidf_matrix.to_html("sample_tfidf_matrix.html")

print("Amostras das matrizes salvas como HTML.")
