# No arquivo TF-IDF.py faça uma função que carregue um corpus sintético de sua preferência e
# retorne a matriz a matriz TF-IDF.
# Faça a implementação manual dos cálculos.

import pandas as pd
import numpy as np
import math

def tf(termo, documento):
    palavras = documento.split()
    cont_termo = 0
    for palavra in palavras:
        if palavra.lower() == termo.lower():
            cont_termo += 1
    total_termos = len(palavras)
    return cont_termo / total_termos

def idf(termo, corpus):
    num_documentos_com_termo = 0
    for documento in corpus:
        if termo.lower() in documento.lower().split():
            num_documentos_com_termo += 1
    if num_documentos_com_termo > 0:
        return math.log(len(corpus) / num_documentos_com_termo)
    else:
        return 0.0

def tf_idf(corpus):
    termos_unicos = list(set(termo.lower() for documento in corpus for termo in documento.split()))

    matriz_tf_idf = np.zeros((len(corpus), len(termos_unicos)))

    for i, documento in enumerate(corpus):
        for j, termo in enumerate(termos_unicos):
            valor_tf = tf(termo, documento)
            valor_idf = idf(termo, corpus)
            matriz_tf_idf[i, j] = valor_tf * valor_idf

    matriz_tf_idf_df = pd.DataFrame(matriz_tf_idf, columns=termos_unicos)
    return matriz_tf_idf_df

corpus = [
    "A vida com python é uma linguagem com tipificação fraca",
    "O python é de alto nível, isso significa que está mais próxima da linguagem humana",
    "A linguagem python se chama assim por causa de um programa que o seu criador gostava",
    "Não sei mais o que escrever"
]

matriz_tf_idf = tf_idf(corpus)
print(matriz_tf_idf)
