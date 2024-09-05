import numpy as np
import pandas as pd
import math
from collections import Counter

def calcular_tf(documento):
    palavras = documento.split()
    contagem_termos = Counter(palavras)
    qtde_termos = len(palavras)
    valores_tf = {termo: contagem / qtde_termos for termo, contagem in contagem_termos.items()}
    return valores_tf

def calcular_idf(corpus):
    qtde_documentos = len(corpus)
    valores_idf = {}
    todas_palavras = set(palavra for documento in corpus for palavra in documento.split())

    for palavra in todas_palavras:
        qtde_documentos_com_termo = sum(1 for documento in corpus if palavra in documento.split())
        valores_idf[palavra] = math.log(qtde_documentos / qtde_documentos_com_termo) if qtde_documentos_com_termo > 0 else 0.0

    return valores_idf

def calcular_tf_idf(corpus):
    valores_idf = calcular_idf(corpus)
    matriz_tf_idf = []

    for documento in corpus:
        valores_tf = calcular_tf(documento)
        valores_tf_idf = {termo: valores_tf.get(termo, 0) * valores_idf[termo] for termo in valores_idf}
        matriz_tf_idf.append(valores_tf_idf)

    # Criar um DataFrame
    df_tf_idf = pd.DataFrame(matriz_tf_idf).fillna(0)
    return df_tf_idf


corpus = ["A lua brilha no céu", "O vento sopra forte"]

print(calcular_tf_idf(corpus))