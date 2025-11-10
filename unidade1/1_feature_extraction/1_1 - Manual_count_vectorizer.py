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
    #tokenizar
    palavras_unicas = list(get_vocabulario(texto))
    #criar saco de palavras (id para cada termo)
    saco = saco_palavras(palavras_unicas)
    #criar a matriz de dicionario para cada sentença
    matriz = pd.DataFrame(np.zeros((len(texto_list), len(palavras_unicas))), columns=palavras_unicas, index=texto_list)
    # print(matriz)
    for idx, row in matriz.iterrows():# pra cada indice e linhas
        for palavra in tokenizer(idx):
            row[palavra] += 1
    
    for col in matriz.columns:
        matriz[col] = matriz[col].astype('int')

    matriz.index = range(len(matriz))
    
    return matriz 

# print(df['review'].head())

# matriz = count_vectorizer(df['review'])
# matriz.to_csv('matriz_count_vectorizer.xlsx')

# print(df['review'].iloc[0])


# frases = ['A fruta é vermelha e é uma fruta', 'A fruta é verde', 'Gatos são legais', 'Sete maçãs', 'A maçã é vermelha', 'A caneta tem tinta vermelha', 'A maçã e a caneta possuem cor vermelha']
frases = ['A maçã é vermelha', 'A caneta tem tinta vermelha', 'A maçã e a caneta possuem cor vermelha']

df_teste = pd.DataFrame({'frases': frases})

df_teste['frases'] = df_teste['frases'].str.lower()

print(count_vectorizer(df_teste['frases']))

cv = CountVectorizer()

X = cv.fit_transform(df_teste['frases'])

# Transforma a matriz esparsa em um DataFrame denso
df_vetor = pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out())

print(df_vetor)