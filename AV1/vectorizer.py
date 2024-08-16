import pandas as pd
import numpy as np
from collections import Counter
from preprocessing import preprocess_text


def count_vectorizer(corpus, vocab=None, max_features=1000):
    if vocab is None:
        vocab = Counter()
        for doc in corpus:
            tokens = doc.split()
            vocab.update(tokens)
        vocab = dict(vocab.most_common(max_features))

    vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))

    count_matrix = np.zeros((len(corpus), len(vocab)), dtype=np.float32)

    for i, doc in enumerate(corpus):
        tokens = doc.split()
        for token in tokens:
            if token in vocab:
                count_matrix[i, list(vocab.keys()).index(token)] += 1

    return count_matrix, vocab


def tfidf_vectorizer(corpus, vocab=None, max_features=1000):
    count_matrix, vocab = count_vectorizer(corpus, vocab, max_features)
    n_docs = len(corpus)
    idf = {}

    for word in vocab:
        doc_count = (count_matrix[:, list(vocab.keys()).index(word)] > 0).sum()
        idf[word] = np.log((n_docs + 1) / (doc_count + 1)) + 1

    tfidf_matrix = count_matrix.copy()
    for word in vocab:
        col_index = list(vocab.keys()).index(word)
        # Evitar divisão por zero
        tfidf_matrix[:, col_index] = np.divide(
            count_matrix[:, col_index],
            count_matrix.sum(axis=1),
            out=np.zeros_like(count_matrix[:, col_index], dtype=np.float32),
            where=(count_matrix.sum(axis=1) != 0)
        ) * idf[word]

    return tfidf_matrix, vocab


# Teste
if __name__ == "__main__":
    file_path = 'C:/Users/laura/Downloads/buscape_processado.csv'
    data = pd.read_csv(file_path)

    if 'review_text_processed' not in data.columns:
        raise ValueError("O arquivo CSV deve conter a coluna 'review_text_processed'.")

    data['review_text_processed'] = data['review_text_processed'].fillna('').astype(str)

    preprocessed_data = [preprocess_text(doc) for doc in data['review_text_processed']]

    count_matrix, vocab = count_vectorizer(preprocessed_data)
    print("Matriz de Contagem de Palavras:")
    print(pd.DataFrame(count_matrix, columns=vocab.keys()))

    tfidf_matrix, vocab = tfidf_vectorizer(preprocessed_data)
    print("\nMatriz TF-IDF:")
    print(pd.DataFrame(tfidf_matrix, columns=vocab.keys()))
