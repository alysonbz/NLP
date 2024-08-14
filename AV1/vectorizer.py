import numpy as np
from collections import defaultdict


# Função manual de CountVectorizer
def count_vectorizer(corpus):
    word_dict = defaultdict(int)
    for text in corpus:
        for word in text.split():
            word_dict[word] += 1

    vocabulary = {word: i for i, word in enumerate(word_dict.keys())}

    vectorized_corpus = []
    for text in corpus:
        vector = np.zeros(len(vocabulary))
        for word in text.split():
            if word in vocabulary:
                vector[vocabulary[word]] += 1
        vectorized_corpus.append(vector)

    return np.array(vectorized_corpus), vocabulary


# Função manual de TF-IDF
def tfidf_vectorizer(corpus):
    word_dict = defaultdict(int)
    doc_count = defaultdict(int)
    num_docs = len(corpus)

    for text in corpus:
        seen_words = set()
        for word in text.split():
            word_dict[word] += 1
            if word not in seen_words:
                doc_count[word] += 1
                seen_words.add(word)

    vocabulary = {word: i for i, word in enumerate(word_dict.keys())}

    vectorized_corpus = []
    for text in corpus:
        vector = np.zeros(len(vocabulary))
        for word in text.split():
            if word in vocabulary:
                tf = text.split().count(word) / len(text.split())
                idf = np.log(num_docs / (doc_count[word] + 1))
                vector[vocabulary[word]] = tf * idf
        vectorized_corpus.append(vector)

    return np.array(vectorized_corpus), vocabulary
