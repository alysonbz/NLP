import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureExtractor:
    def __init__(self, corpus):
        self.corpus = corpus

    def statistical_analysis(self):
        word_counts = [len(text.split()) for text in self.corpus]
        stats = {
            "mean_length": np.mean(word_counts),
            "max_length": np.max(word_counts),
            "min_length": np.min(word_counts),
            "total_tokens": np.sum(word_counts),
            "var_tokens": np.var(word_counts)
        }
        return stats

    def count_vectorizer(self):
        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(self.corpus)
        return count_matrix, vectorizer.get_feature_names_out()

    def tfidf_vectorizer(self):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.corpus)
        return tfidf_matrix, vectorizer.get_feature_names_out()

    def cooccurrence_matrix(self, window_size=2):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.corpus)
        vocab = vectorizer.get_feature_names_out()
        cooccurrence = (X.T @ X).toarray()
        np.fill_diagonal(cooccurrence, 0)  # Remover autocorrelações
        return pd.DataFrame(cooccurrence, index=vocab, columns=vocab)

    def word2vec(self, vector_size=100, window=5, min_count=1, epochs=10):
        tokenized_corpus = [text.split() for text in self.corpus]
        model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count
        )
        model.train(tokenized_corpus, total_examples=len(tokenized_corpus), epochs=epochs)
        return model
