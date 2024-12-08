import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
from gensim.models import Word2Vec

dataset = pd.read_csv('C:/Users/Viana/PycharmProjects/savio/NLP/AV1/processed_dataset.csv')

class FeatureExtraction:
    def __init__(self, dataset: pd.DataFrame, text_column: str = "clean_title"):
        self.dataset = dataset
        self.text_column = text_column

    def statistical_analysis(self) -> pd.DataFrame:

        stats = self.dataset[self.text_column].apply(lambda x: {
            "word_count": len(x.split()),
            "avg_word_length": np.mean([len(word) for word in x.split()]),
            "char_count": len(x)
        })

        # Converter para DataFrame
        stats_df = pd.DataFrame(stats.tolist())
        self.dataset = pd.concat([self.dataset, stats_df], axis=1)
        return self.dataset

    def count_vectorizer(self, max_features=500):
        vectorizer = CountVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(self.dataset[self.text_column])
        feature_names = vectorizer.get_feature_names_out()

        return pd.DataFrame(X.toarray(), columns=feature_names)

    def tfidf_vectorizer(self, max_features=500):
        tfidf = TfidfVectorizer(max_features=max_features)
        X = tfidf.fit_transform(self.dataset[self.text_column])
        feature_names = tfidf.get_feature_names_out()

        return pd.DataFrame(X.toarray(), columns=feature_names)

    def cooccurrence_matrix(self, window_size=2):
        # Obter todas as palavras
        words = [text.split() for text in self.dataset[self.text_column]]
        word_list = [word for sentence in words for word in sentence]

        # Construir a matriz de coocorrência
        cooccurrence = Counter()
        for sentence in words:
            for i, word in enumerate(sentence):
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, len(sentence))
                for j in range(start, end):
                    if i != j:
                        cooccurrence[(word, sentence[j])] += 1

        # Converter para DataFrame
        vocab = list(set(word_list))
        co_matrix = pd.DataFrame(0, index=vocab, columns=vocab)

        for (word1, word2), count in cooccurrence.items():
            co_matrix.loc[word1, word2] = count

        return co_matrix

    def word2vec(self, vector_size=100, window=5, min_count=1, workers=4):
        sentences = [text.split() for text in self.dataset[self.text_column]]
        model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

        word_vectors = {word: model.wv[word] for word in model.wv.index_to_key}
        return word_vectors


# Instanciar a classe
extractor = FeatureExtraction(dataset)

# Análise estatística
dataset_stats = extractor.statistical_analysis()
print("Análise Estatística:\n", dataset_stats.head())

# CountVectorizer
count_matrix = extractor.count_vectorizer()
print("Count Vectorizer:\n", count_matrix.head())

# TF-IDF
tfidf_matrix = extractor.tfidf_vectorizer()
print("TF-IDF:\n", tfidf_matrix.head())

# Matriz de Coocorrência
cooccurrence_matrix = extractor.cooccurrence_matrix()
print("Matriz de Coocorrência:\n", cooccurrence_matrix.head())

# Word2Vec
word_vectors = extractor.word2vec()
print("Word2Vec (vetores de palavras):")
for word, vector in list(word_vectors.items())[:5]:
     print(f"{word}: {vector[:5]}")  # Mostrando os 5 primeiros valores de cada vetor

