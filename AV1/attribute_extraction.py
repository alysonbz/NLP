from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from gensim.models import KeyedVectors
import string
from spellchecker import SpellChecker
import re
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

"""Voce deve implementar funções para extração de atributos com Analise estatística
individual, CountVectorizer, TF-IDF, matriz de coocorrência e word2vec..
Faça uma função para cada forma de extração de atributo, sugiro que seja construída
uma classe para essas funções. A estrutura do código deve permitir que possam ser
importadas as funções em outras questões"""

class ExtratorManual:
    def __init__(self):
        pass

    def criar_df_atributos(self, df):
        df_attr = pd.DataFrame()
        df_attr['caracteres'] = df['essay'].apply(self.contar_caracteres)
        df_attr['palavras'] = df['essay'].apply(self.contar_palavras)
        df_attr['frases'] = df['essay'].apply(self.contar_frases)
        df_attr['pontuacao'] = df['essay'].apply(self.contar_pontuacao)
        df_attr['digitos'] = df['essay'].apply(self.contar_digitos)
        df_attr['quebras_linhas'] = df['essay'].apply(self.contar_quebras_linhas)
        df_attr['palavras_corretas'] = df['essay'].apply(self.contar_palavras_corretas)
        return df_attr

    def contar_palavras(self, texto):
        return len(texto.split())

    def contar_caracteres(self, texto):
        return len(texto)

    def contar_frases(self, texto):
        return len(texto.split('.'))

    def contar_pontuacao(self, texto):
        return len([char for char in texto if char in string.punctuation])

    def contar_digitos(self, texto):
        return len([char for char in texto if char.isdigit()])

    def contar_quebras_linhas(self, texto):
        return texto.count('\n')

    def contar_palavras_corretas(self, texto):
        sp = SpellChecker(language='pt')
        # tokenizador simples que mantém apenas letras
        palavras = re.findall(r'[a-zA-Zà-úÀ-ÚçÇ]+', texto)
        desconhecidas = sp.unknown(palavras)
        return len(palavras) - len(desconhecidas)

class Extratores:

    def __init__(self, text_col="essay",
                 cv_params=None, tfidf_params=None):

        self.text_col = text_col

        # parâmetros opcionais
        self.cv = CountVectorizer(**(cv_params or {}))
        self.tfidf = TfidfVectorizer(**(tfidf_params or {}))

        # modelo de embeddings opcional
        repo = "nilc-nlp/word2vec-cbow-50d"

        # Baixa o arquivo de vetores
        embeddings_path = hf_hub_download(
            repo_id=repo,
            filename="embeddings.safetensors"
        )

        # Carrega os vetores como numpy
        data = load_file(embeddings_path)
        vectors = data["embeddings"]      

        # Baixa o vocabulário
        vocab_path = hf_hub_download(
            repo_id=repo,
            filename="vocab.txt"
        )

        # Lê o vocabulário
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f]

        # Criar KeyedVectors e carregar manualmente
        kv = KeyedVectors(vector_size=vectors.shape[1])
        kv.add_vectors(vocab, vectors)

        # Guardar modelo interno
        self.embedding_model = kv

    # ------- BoW -------
    def cv_fit_transform(self, df):
        textos = df[self.text_col].astype(str).values
        return self.cv.fit_transform(textos)

    def cv_transform(self, df):
        textos = df[self.text_col].astype(str).values
        return self.cv.transform(textos)

    # ------- TF-IDF -------
    def tfidf_fit_transform(self, df):
        textos = df[self.text_col].astype(str).values
        return self.tfidf.fit_transform(textos)

    def tfidf_transform(self, df):
        textos = df[self.text_col].astype(str).values
        return self.tfidf.transform(textos)

    # ------- Embeddings (Word2Vec / FastText) -------
    def embedding_vector(self, text):
        """Retorna o vetor médio do documento."""
        tokens = str(text).lower().split()

        vetores = [
            self.embedding_model[word]
            for word in tokens
            if word in self.embedding_model
        ]

        if not vetores:
            return np.zeros(self.embedding_model.vector_size)

        return np.mean(vetores, axis=0)

    def embedding_transform(self, df):
        """Transforma um DataFrame inteiro em matriz de embeddings."""
        return np.vstack(df[self.text_col].apply(self.embedding_vector))