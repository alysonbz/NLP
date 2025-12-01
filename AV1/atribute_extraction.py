import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec


# ======================================================================
# Classe para Extração de Atributos
# ======================================================================

class AttributeExtractor:

    def __init__(self, corpus):
        """
        corpus: lista de textos pré-processados (strings)
        """
        self.corpus = corpus
        self.vectorizer_count = None
        self.vectorizer_tfidf = None
        self.word2vec_model = None

    # ===============================================================
    # 1. Análise estatística individual
    # ===============================================================

    def statistical_analysis(self):
        """
        Retorna análise estatística do corpus:
        - tamanho do vocabulário
        - top 20 palavras
        - número total de tokens
        """
        tokens = " ".join(self.corpus).split()
        freq = Counter(tokens)

        return {
            "vocab_size": len(freq),
            "top_20_words": freq.most_common(20),
            "total_tokens": len(tokens),
        }

    # ===============================================================
    # 2. CountVectorizer (Bag-of-Words)
    # ===============================================================

    def count_vectorizer(self):
        """
        Retorna matriz Bag-of-Words e o vetorizer treinado.
        """
        self.vectorizer_count = CountVectorizer()
        X = self.vectorizer_count.fit_transform(self.corpus)
        return X, self.vectorizer_count

    # ===============================================================
    # 3. TF-IDF
    # ===============================================================

    def tfidf_vectorizer(self):
        """
        Retorna matriz TF-IDF e o vetorizer treinado.
        """
        self.vectorizer_tfidf = TfidfVectorizer()
        X = self.vectorizer_tfidf.fit_transform(self.corpus)
        return X, self.vectorizer_tfidf

    # ===============================================================
    # 4. Matriz de Coocorrência
    # ===============================================================

    def cooccurrence_matrix(self, window_size=2):
        """
        Constrói uma matriz de coocorrência usando janela deslizante.
        """
        vocab = list(set(" ".join(self.corpus).split()))
        vocab_index = {word: idx for idx, word in enumerate(vocab)}

        # matriz vazia
        matrix = np.zeros((len(vocab), len(vocab)), dtype=np.int32)

        # preenche matriz
        for text in self.corpus:
            tokens = text.split()
            for i in range(len(tokens)):
                target = tokens[i]
                left = max(i - window_size, 0)
                right = min(i + window_size + 1, len(tokens))

                for j in range(left, right):
                    if i != j:
                        context = tokens[j]
                        matrix[vocab_index[target]][vocab_index[context]] += 1

        return matrix, vocab_index

    # ===============================================================
    # 5. Word2Vec (Gensim)
    # ===============================================================

    def word2vec(self, vector_size=100, window=5, min_count=1, epochs=30):
        """
        Treina Word2Vec e retorna:
        - embeddings como matriz
        - palavras do vocabulário
        - modelo treinado
        """
        tokenized = [text.split() for text in self.corpus]

        self.word2vec_model = Word2Vec(
            sentences=tokenized,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1,
            epochs=epochs
        )

        words = list(self.word2vec_model.wv.index_to_key)
        embeddings = np.array([self.word2vec_model.wv[word] for word in words])

        return embeddings, words, self.word2vec_model


# ======================================================================
# Execução direta (opcional para teste)
# ======================================================================
if __name__ == "__main__":

    # importar preprocessamento
    from preprocessing import preprocess_dataset

    # carregar dataset
    df = pd.read_parquet("train-00000-of-00001.parquet")

    # aplicar preprocessamento antes da extração
    df_clean = preprocess_dataset(df)

    # criar corpus unindo todas as sentenças limpas
    corpus = df_clean["premise_clean"].tolist() + df_clean["hypothesis_clean"].tolist()

    # iniciar extrator
    extractor = AttributeExtractor(corpus)

    print("\n========== ANÁLISE ESTATÍSTICA ==========")
    print(extractor.statistical_analysis())

    print("\n========== COUNT VECTORIZER ==========")
    bow, vect = extractor.count_vectorizer()
    print("Shape:", bow.shape)

    print("\n========== TF-IDF ==========")
    tfidf, vect2 = extractor.tfidf_vectorizer()
    print("Shape:", tfidf.shape)

    print("\n========== MATRIZ DE COOOCORRÊNCIA ==========")
    cooc, vocab = extractor.cooccurrence_matrix(window_size=2)
    print("Matriz:", cooc.shape)

    print("\n========== WORD2VEC ==========")
    emb, words, model = extractor.word2vec()
    print("Embeddings:", emb.shape)
