import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec


class ExtratorAtributos:

    def __init__(self, textos_processados):
        
        self.textos_tokens = textos_processados
        self.textos_string = [" ".join(tokens) for tokens in textos_processados]

        self.cv_vectorizer = None
        self.tfidf_vectorizer = None
        self.w2v_model = None

    # ANÁLISE ESTATÍSTICA

    def analise_estatistica(self):
        estatisticas = []

        for tokens in self.textos_tokens:

            if len(tokens) == 0:
                estatisticas.append({
                    "n_tokens": 0,
                    "n_unicos": 0,
                    "media_tamanho": 0,
                    "n_caracteres": 0,
                    "n_simbolos": 0
                })
                continue

            tamanhos = [len(t) for t in tokens]

            estatisticas.append({
                "n_tokens": len(tokens),
                "n_unicos": len(set(tokens)),
                "media_tamanho": np.mean(tamanhos),
                "n_caracteres": sum(tamanhos),
                "n_simbolos": sum(sum(1 for c in t if not c.isalnum()) for t in tokens)
            })

        return pd.DataFrame(estatisticas)

    # COUNT VECTORIZER
    def cv_fit_transform(self, max_features=500):
        """
        Treina o CountVectorizer no conjunto de treino.
        """
        self.cv_vectorizer = CountVectorizer(max_features=max_features)
        matriz = self.cv_vectorizer.fit_transform(self.textos_string)
        return matriz.toarray()

    def cv_transform(self):
        """
        Aplica no conjunto de teste (já treinado).
        """
        if self.cv_vectorizer is None:
            raise ValueError("Erro: chame cv_fit_transform() antes de cv_transform().")

        matriz = self.cv_vectorizer.transform(self.textos_string)
        return matriz.toarray()

    # TF-IDF

    def tfidf_fit_transform(self, max_features=500):
        """
        Treina o TF-IDF no conjunto de treino.
        """
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        matriz = self.tfidf_vectorizer.fit_transform(self.textos_string)
        return matriz.toarray()

    def tfidf_transform(self):
        """
        Aplica no conjunto de teste.
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("Erro: chame tfidf_fit_transform() antes de tfidf_transform().")

        matriz = self.tfidf_vectorizer.transform(self.textos_string)
        return matriz.toarray()

    # WORD2VEC (treinado do zero)

    def word2vec_fit(self, vector_size=100, window=5, min_count=1, workers=4):
        """
        Treina o modelo Word2Vec nos textos tokenizados do conjunto de treino.
        """
        self.w2v_model = Word2Vec(
            sentences=self.textos_tokens,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )
        return self.w2v_model

    def _doc_vector(self, tokens):
        """
        Gera vetor médio de um documento.
        """
        if self.w2v_model is None:
            raise ValueError("Erro: Word2Vec ainda não foi treinado. Use word2vec_fit().")

        vetores = [
            self.w2v_model.wv[w]
            for w in tokens
            if w in self.w2v_model.wv
        ]

        if len(vetores) == 0:
            return np.zeros(self.w2v_model.vector_size)

        return np.mean(vetores, axis=0)

    def word2vec_transform(self):
        if self.w2v_model is None:
            raise ValueError("Erro: Word2Vec ainda não foi treinado. Use word2vec_fit().")

        matriz = np.array([self._doc_vector(tokens) for tokens in self.textos_tokens])
        return matriz
