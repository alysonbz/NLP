# feature_extraction.py
# Funções de extração de atributos para textos

import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec


class FeatureExtractor:

    def __init__(self, texts):
        """
        texts: lista de textos já pré-processados
        """
        self.texts = [t if isinstance(t, str) else "" for t in texts]

    # Análise estatística individual
    def statistical_analysis(self):
      stats = {
          "num_tokens": [],
          "num_unique_tokens": [],
          "avg_token_size": [],
      }

      for text in self.texts:
          tokens = text.split()
          if len(tokens) == 0:
              stats["num_tokens"].append(0)
              stats["num_unique_tokens"].append(0)
              stats["avg_token_size"].append(0)
              continue

          stats["num_tokens"].append(len(tokens))
          stats["num_unique_tokens"].append(len(set(tokens)))
          avg = np.mean([len(t) for t in tokens])
          stats["avg_token_size"].append(avg)

      df_stats = pd.DataFrame(stats)
      return df_stats.values


    # Bag of Words
    def bag_of_words(self, max_features=2000, ngram_range=(1,1)):
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        matrix = vectorizer.fit_transform(self.texts)
        return matrix, vectorizer


    # TF-IDF
    def tfidf(self, max_features=2000, ngram_range=(1,1)):
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        matrix = vectorizer.fit_transform(self.texts)
        return matrix, vectorizer


    # Coocorrência
    def cooccurrence_matrix(self, window_size=2):

        cooc_list = []

        for text in self.texts:
            tokens = text.split()
            cooc_counts = Counter()

            for i, w in enumerate(tokens):
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)

                for j in range(start, end):
                    if i == j:
                        continue
                    pair = tuple(sorted([w, tokens[j]]))
                    cooc_counts["_".join(pair)] += 1

            cooc_list.append(cooc_counts)

        # Criar vocabulário global
        all_keys = set()
        for c in cooc_list:
            all_keys.update(c.keys())

        all_keys = sorted(all_keys)

        matrix = np.zeros((len(self.texts), len(all_keys)), dtype=np.float32)

        for i, counts in enumerate(cooc_list):
            for j, key in enumerate(all_keys):
                matrix[i, j] = counts.get(key, 0)

        return matrix, all_keys


    # 5. Word2Vec
    def word2vec(self, vector_size=100, window=5, min_count=1, epochs=15):

        tokenized = [t.split() for t in self.texts]

        model = Word2Vec(
            sentences=tokenized,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1,
            epochs=epochs
        )

        doc_vectors = []

        for tokens in tokenized:
            v = [model.wv[w] for w in tokens if w in model.wv]
            if len(v) > 0:
                doc_vectors.append(np.mean(v, axis=0))
            else:
                doc_vectors.append(np.zeros(vector_size))

        return np.array(doc_vectors), model