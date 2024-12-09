import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureExtractor:

    def __init__(self, corpus):
        self.corpus = corpus

    def statistical_analysis(self):
        """
        Realiza uma análise estatística básica sobre o comprimento das palavras.
        - Média, máximo, mínimo, total de tokens e variância do número de tokens por texto.
        """
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
        """
        Aplica o CountVectorizer, que cria uma matriz de contagem de palavras.
        Retorna a matriz de contagem e o vocabulário (lista de palavras).
        """
        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(self.corpus)
        return count_matrix, vectorizer.get_feature_names_out()

    def tfidf_vectorizer(self):
        """
        Aplica o TF-IDF Vectorizer, que transforma o texto em uma matriz TF-IDF.
        Retorna a matriz TF-IDF e o vocabulário (lista de palavras).
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.corpus)
        return tfidf_matrix, vectorizer.get_feature_names_out()

    def cooccurrence_matrix(self, window_size=2):
        """
         Cria uma matriz de coocorrência baseada na contagem de palavras.
         - A coocorrência é calculada com base em uma janela de palavras especificada (default = 2).
         """
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.corpus)
        vocab = vectorizer.get_feature_names_out()
        cooccurrence = (X.T @ X).toarray()
        np.fill_diagonal(cooccurrence, 0)  # Remover autocorrelações
        return pd.DataFrame(cooccurrence, index=vocab, columns=vocab)

    def word2vec(self, vector_size=100, window=5, min_count=1, epochs=10):
        """
        Aplica o modelo Word2Vec do gensim para aprender embeddings de palavras.
        - Retorna um modelo Word2Vec treinado com o corpus tokenizado.
        """
        tokenized_corpus = [text.split() for text in self.corpus]
        model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count
        )
        model.train(tokenized_corpus, total_examples=len(tokenized_corpus), epochs=epochs)
        return model

