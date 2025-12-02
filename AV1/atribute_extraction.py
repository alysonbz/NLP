"""
Voce deve implementar funções para extração de atributos com Analise estatística individual,
CountVectorizer, TF-IDF, matriz de coocorrência e word2vec.. Faça uma função para cada forma 
de extração de atributo, sugiro que seja construída uma classe para essas funções.
A estrutura do código deve permitir que possam ser importadas as funções em outras questões.
"""

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
        self.vocab = None

    # --------------------------
    #    ANÁLISE ESTATÍSTICA
    # --------------------------

    def analise_estatistica(self):
        estatisticas = []

        for tokens in self.textos_tokens:

            if not tokens:
                estatisticas.append({
                    "n_tokens": 0,
                    "n_unicos": 0,
                    "media_tamanho": 0,
                    "n_caracteres": 0,
                    "n_simbolos": 0
                })
                continue

            tamanhos = np.array([len(t) for t in tokens])

            estatisticas.append({
                "n_tokens": len(tokens),
                "n_unicos": len(set(tokens)),
                "media_tamanho": tamanhos.mean(),
                "n_caracteres": tamanhos.sum(),
                "n_simbolos": sum(c for t in tokens for c in map(lambda x: not x.isalnum(), t))
            })

        return pd.DataFrame(estatisticas)

    # --------------------------
    #      COUNT VECTORIZER
    # --------------------------

    def cv_fit_transform(self, max_features=500):
        self.cv_vectorizer = CountVectorizer(max_features=max_features)
        return self.cv_vectorizer.fit_transform(self.textos_string).toarray()

    def cv_transform(self, novos_textos=None):
        if self.cv_vectorizer is None:
            raise ValueError("Use cv_fit_transform() antes.")

        if novos_textos is None:
            textos = self.textos_tokens
        else:
            textos = novos_textos

        textos_str = [" ".join(t) for t in textos]
        return self.cv_vectorizer.transform(textos_str).toarray()

    # --------------------------
    #          TF-IDF
    # --------------------------

    def tfidf_fit_transform(self, max_features=500):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        return self.tfidf_vectorizer.fit_transform(self.textos_string).toarray()


    def tfidf_transform(self, novos_textos=None):
        if self.tfidf_vectorizer is None:
            raise ValueError("Use tfidf_fit_transform() antes.")

        if novos_textos is None:
            textos = self.textos_tokens
        else:
            textos = novos_textos

        novos_str = [" ".join(t) for t in textos]
        return self.tfidf_vectorizer.transform(novos_str).toarray()

    # --------------------------
    #       WORD2VEC TREINO
    # --------------------------

    def word2vec_fit(self, vector_size=100, window=5, min_count=1, workers=4):
        self.w2v_model = Word2Vec(
            sentences=self.textos_tokens,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        ).wv
        
        return self.w2v_model

    # --------------------------
    #       DOC VECTOR
    # --------------------------

    def _doc_vector(self, tokens):
        vetores = [self.w2v_model[w] for w in tokens if w in self.w2v_model]

        if not vetores:
            return np.zeros(self.w2v_model.vector_size)

        return np.mean(vetores, axis=0)

    def word2vec_transform(self, novos_textos=None):
        if self.w2v_model is None:
            raise ValueError("Treine o modelo Word2Vec com word2vec_fit().")

        if novos_textos is None:
            tokens_lista = self.textos_tokens
        else:
            tokens_lista = novos_textos

        return np.array([self._doc_vector(tokens) for tokens in tokens_lista])