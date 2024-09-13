import pandas as pd
import math
from collections import defaultdict
import re

def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)  # Remove números
    texto = re.sub(r'[^\w\s]', '', texto)  # Remove pontuação
    return texto

def calcular_frequencia_termo(documento):
    palavras = preprocessar_texto(documento).split()
    total_palavras = len(palavras)
    contagem_palavras = defaultdict(int)

    for palavra in palavras:
        contagem_palavras[palavra] += 1

    tf = {palavra: contagem / total_palavras for palavra, contagem in contagem_palavras.items()}
    return tf

def calcular_idf(corpus):
    num_documentos = len(corpus)
    frequencia_documentos = defaultdict(int)

    for doc in corpus:
        palavras_unicas = set(preprocessar_texto(doc).split())
        for palavra in palavras_unicas:
            frequencia_documentos[palavra] += 1

    idf = {}
    for palavra, freq in frequencia_documentos.items():
        idf[palavra] = math.log((num_documentos + 1) / (freq + 1))  # +1 para evitar zero

    return idf

def criar_matriz_tf_idf(corpus):
    idf = calcular_idf(corpus)
    matriz_tf_idf = []

    for doc in corpus:
        tf = calcular_frequencia_termo(doc)
        tf_idf = {palavra: tf.get(palavra, 0) * idf.get(palavra, 0) for palavra in idf}
        matriz_tf_idf.append(tf_idf)

    df_tf_idf = pd.DataFrame(matriz_tf_idf).fillna(0)
    return df_tf_idf

# Exemplo de corpus
corpus_exemplo = [
    "A universidade deveria criar programas de apoio à maternidade que ajudam as discentes na adaptação à nova fase da vida.",
    "Os programas de apoio à maternidade são importantes para garantir o sucesso acadêmico das mães."
]

if __name__ == '__main__':
    df_resultado_tf_idf = criar_matriz_tf_idf(corpus_exemplo)
    print(df_resultado_tf_idf)
