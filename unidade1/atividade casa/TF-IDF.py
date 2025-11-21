import pandas as pd
import math


def carregar_corpus_sintetico():
    corpus = [
        'o filme foi lindo e realmente fantastico',
        'o filme era um terror',
        'o anime estava otimo no inicio, depois ficou horrivel'
    ]
    return corpus


def calcular_tf(tokens_documento):
    tf = {}
    total_palavras = len(tokens_documento)

    for palavra in tokens_documento:
        tf[palavra] = tf.get(palavra, 0) + 1

    for palavra in tf:
        tf[palavra] = tf[palavra] / total_palavras

    return tf


def calcular_idf(lista_tokens):
    total_docs = len(lista_tokens)
    vocabulario = set(p for doc in lista_tokens for p in doc)

    idf = {}

    for palavra in vocabulario:
        docs_contem = sum(1 for doc in lista_tokens if palavra in doc)
        idf[palavra] = math.log((total_docs + 1) / (docs_contem + 1)) + 1

    return idf


def calcular_tfidf(corpus):
    lista_tokens = [doc.lower().split() for doc in corpus]
    lista_tf = [calcular_tf(tokens) for tokens in lista_tokens]
    idf = calcular_idf(lista_tokens)

    vocabulario = sorted(idf.keys())
    matriz_tfidf = []

    for tf_doc in lista_tf:
        linha = [tf_doc.get(palavra, 0) * idf[palavra] for palavra in vocabulario]
        matriz_tfidf.append(linha)

    return pd.DataFrame(matriz_tfidf, columns=vocabulario)


if __name__ == '__main__':
    corpus = carregar_corpus_sintetico()
    matriz_tfidf = calcular_tfidf(corpus)
    print('\nCorpus atualizado:')

    for frase in corpus:
        print(frase)

    print(f'\nMatriz TF-IDF:\n{matriz_tfidf}')