import numpy as np
import pandas as pd
from collections import Counter
from math import log

def calculate_tf_idf(corpus):
    tokenized_sentences = [sentence.lower().split() for sentence in corpus]
    unique_words = list(set(word for sentence in tokenized_sentences for word in sentence))  # Corrigido para lista

    tf_matrix = pd.DataFrame(0, index=unique_words, columns=range(len(corpus)), dtype=float)
    idf_values = {}

    for i, sentence in enumerate(tokenized_sentences):
        word_counts = Counter(sentence)
        total_words = sum(word_counts.values())
        for word in word_counts:
            tf_matrix.loc[word, i] = word_counts[word] / total_words

    total_sentences = len(corpus)
    for word in unique_words:
        containing_sentences = sum(1 for sentence in tokenized_sentences if word in sentence)
        idf_values[word] = log(total_sentences / (1 + containing_sentences))

    tf_idf_matrix = tf_matrix.copy()
    for word in idf_values:
        tf_idf_matrix.loc[word] *= idf_values[word]

    return tf_idf_matrix

# Corpus
corpus = [
    "inflation increased unemployment",
    "company increased sales",
    "fear increased pulse"
]

tf_idf_matrix = calculate_tf_idf(corpus)
print(tf_idf_matrix)
