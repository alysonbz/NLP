import pandas as pd
from src.utils import load_movie_review_clean_dataset
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = load_movie_review_clean_dataset()

def tokenizer(string):
    return string.split(' ')

def get_vocabulario(textos: pd.DataFrame):
    todas_palavras = []
    for linha in textos.to_list():
        palavras = tokenizer(linha)
        todas_palavras.extend(palavras)
    return set(todas_palavras)

def saco_palavras(vocab):
    return {k: i for i, k in enumerate(vocab)}

def count_vectorizer(texto: pd.Series) -> pd.DataFrame:
    texto_list = texto.values
    palavras_unicas = sorted(get_vocabulario(texto))  
    matriz = pd.DataFrame(0, index=range(len(texto_list)), columns=palavras_unicas)

    for i, frase in enumerate(texto_list):
        for palavra in tokenizer(frase):
            if palavra in matriz.columns:
                matriz.loc[i, palavra] += 1

    return matriz


frases = ['A maçã é vermelha', 'A caneta tem tinta vermelha', 'A maçã e a caneta possuem cor vermelha']

df_teste = pd.DataFrame({'frases': frases})

df_teste['frases'] = df_teste['frases'].str.lower()

print(count_vectorizer(df_teste['frases']))

cv = CountVectorizer()

X = cv.fit_transform(df_teste['frases'])

df_vetor = pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out())

print(df_vetor)