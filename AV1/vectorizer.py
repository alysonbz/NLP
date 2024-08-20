import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Função para criar uma matriz de contagem (CountVectorizer) manualmente
def manual_count_vectorizer(texts):
    # Construir vocabulário
    vocabulary = set()
    for text in texts:
        vocabulary.update(text.split())
    vocabulary = sorted(vocabulary)
    vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}

    # Criar matriz de contagem
    matrix = np.zeros((len(texts), len(vocabulary)))
    for i, text in enumerate(texts):
        word_count = Counter(text.split())
        for word, count in word_count.items():
            if word in vocab_dict:
                matrix[i, vocab_dict[word]] = count
    return matrix, vocab_dict

# Função para criar uma matriz TF-IDF manualmente
def manual_tfidf_vectorizer(texts):
    matrix, vocab_dict = manual_count_vectorizer(texts)
    num_docs = len(texts)
    term_freq = matrix

    # Calcular a frequência inversa de documentos (IDF)
    doc_freq = np.sum(matrix > 0, axis=0)
    idf = np.log((num_docs + 1) / (doc_freq + 1)) + 1  # +1 para evitar divisão por zero

    # Calcular TF-IDF
    tfidf_matrix = term_freq * idf
    return tfidf_matrix

if __name__ == "__main__":
    # Dados de exemplo
    texts = [
        "A quick brown fox jumps over the lazy dog",
        "The quick brown fox",
        "The lazy dog"
    ]

    # Testar manual_count_vectorizer
    count_matrix, vocab = manual_count_vectorizer(texts)
    print("Count Vectorizer Matrix:")
    print(count_matrix)
    print("Vocabulary:")
    print(vocab)

    # Testar manual_tfidf_vectorizer
    tfidf_matrix = manual_tfidf_vectorizer(texts)
    print("\nTF-IDF Matrix:")
    print(tfidf_matrix)

