import numpy as np
import pandas as pd
import re
from collections import Counter
from typing import List, Tuple
from datasets import load_dataset


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
        if doc:
            words = re.findall(r'\b\w+\b', doc.lower())
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

    vocab_index = {word: i for i, word in enumerate(vocabulary)}
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=int)

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
    matrix, vocabulary = count_vectorizer(documents, vocab=vocab)

    # Calcula TF
    tf_matrix = matrix.astype(float)
    row_sums = tf_matrix.sum(axis=1, keepdims=True)
    np.divide(tf_matrix, row_sums, out=tf_matrix, where=row_sums != 0)

    # Calcula IDF
    num_docs = len(documents)
    doc_freq = np.sum(matrix > 0, axis=0)
    idf = np.log((num_docs + 1) / (doc_freq + 1)) + 1

    # Calcula TF-IDF
    tfidf_matrix = tf_matrix * idf

    return tfidf_matrix, vocabulary


# Carregar o dataset
ds = load_dataset("johnidouglas/twitter-sentiment-pt-BR-md-2-l")
df = ds['train'].to_pandas()

# Extrair textos
num_samples = 10
texts = df['tweet_text'].head(num_samples).tolist()

# Aplicar Count Vectorizer
count_matrix, vocab = count_vectorizer(texts)
print("Matriz de Contagem:\n", count_matrix)
print("Vocabulário:\n", vocab)

# Aplicar TF-IDF Vectorizer
tfidf_matrix, vocab = tfidf_vectorizer(texts)
print("Matriz TF-IDF:\n", tfidf_matrix)

# Visualizar amostras aleatórias das matrizes e vocabulário
random_sample_indices = np.random.choice(count_matrix.shape[0], 5, replace=False)
random_feature_indices = np.random.choice(count_matrix.shape[1], 20, replace=False)

sample_count_matrix = count_matrix[random_sample_indices, :][:, random_feature_indices]
sample_tfidf_matrix = tfidf_matrix[random_sample_indices, :][:, random_feature_indices]
sample_vocab = [vocab[i] for i in random_feature_indices]

df_count_matrix = pd.DataFrame(sample_count_matrix, columns=sample_vocab)
df_tfidf_matrix = pd.DataFrame(sample_tfidf_matrix, columns=sample_vocab)

# Salvar como HTML
df_count_matrix.to_html("sample_count_matrix.html")
df_tfidf_matrix.to_html("sample_tfidf_matrix.html")

print("Amostras das matrizes salvas como HTML.")
