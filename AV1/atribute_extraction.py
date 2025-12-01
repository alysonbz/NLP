from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

class FeatureExtractor:

    def __init__(
        self,
        max_features: Optional[int] = 5000,
        min_df: int = 1,
        max_df: float = 1.0,
        w2v_size: int = 100,
        w2v_window: int = 5,
        w2v_min_count: int = 1,
    ):

        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        self.count_vectorizer: Optional[CountVectorizer] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.word2vec: Optional[Word2Vec] = None

        self.w2v_size = w2v_size
        self.w2v_window = w2v_window
        self.w2v_min_count = w2v_min_count

    def fit_countvectorizer(self, texts: List[str]):
        self.count_vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df
        )
        return self.count_vectorizer.fit_transform(texts)

    def transform_countvectorizer(self, texts: List[str]):
        if self.count_vectorizer is None:
            raise ValueError("CountVectorizer ainda não foi ajustado. Use fit_countvectorizer().")
        return self.count_vectorizer.transform(texts)

    def fit_tfidf(self, texts: List[str]):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df
        )
        return self.tfidf_vectorizer.fit_transform(texts)

    def transform_tfidf(self, texts: List[str]):
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF ainda não foi ajustado. Use fit_tfidf().")
        return self.tfidf_vectorizer.transform(texts)

    def get_word_frequencies(self, texts: List[str]) -> Counter:
        c = Counter()
        for text in texts:
            tokens = text.split()
            c.update(tokens)
        return c

    def build_cooccurrence_matrix(
        self,
        texts: List[str],
        window_size: int = 2
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:

        vocab = {}
        index = 0

        for text in texts:
            for token in text.split():
                if token not in vocab:
                    vocab[token] = index
                    index += 1

        vocab_size = len(vocab)
        matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

        for text in texts:
            tokens = text.split()
            length = len(tokens)
            for i, token in enumerate(tokens):
                token_idx = vocab[token]

                start = max(0, i - window_size)
                end = min(length, i + window_size + 1)

                for j in range(start, end):
                    if i != j:
                        neighbor = tokens[j]
                        neighbor_idx = vocab[neighbor]
                        matrix[token_idx][neighbor_idx] += 1

        df = pd.DataFrame(matrix, index=vocab.keys(), columns=vocab.keys())
        return df, vocab

    def train_word2vec(self, texts: List[str]):
        tokenized = [t.split() for t in texts]
        self.word2vec = Word2Vec(
            sentences=tokenized,
            vector_size=self.w2v_size,
            window=self.w2v_window,
            min_count=self.w2v_min_count,
            workers=4
        )
        return self.word2vec

    def get_word2vec_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.word2vec is None:
            raise ValueError("Word2Vec ainda não foi treinado. Use train_word2vec().")

        embeddings = []
        for text in texts:
            tokens = text.split()
            vectors = []

            for t in tokens:
                if t in self.word2vec.wv:
                    vectors.append(self.word2vec.wv[t])

            if len(vectors) == 0:
                vectors.append(np.zeros(self.w2v_size))

            embeddings.append(np.mean(vectors, axis=0))

        return np.vstack(embeddings)
