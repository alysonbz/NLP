# Packages -------------------------------------------------------------------------------------------------------------

import numpy as np


# Funções de vetorização -----------------------------------------------------------------------------------------------

def calcular_frequencias(docs):
  vocabulario = set()
  frequencias = []
  for doc in docs:
    palavras = doc.split()
    vocabulario.update(palavras)
    freq = {}
    for palavra in palavras:
      freq[palavra] = freq.get(palavra, 0) + 1
    frequencias.append(freq)
  return vocabulario, frequencias


def vetorizar_countvectorizer(docs):
  vocabulario, frequencias = calcular_frequencias(docs)
  vocabulario = list(vocabulario)
  matriz_contagem = []
  for freq in frequencias:
    vetor = [freq.get(palavra, 0) for palavra in vocabulario]
    matriz_contagem.append(vetor)
  return matriz_contagem, vocabulario


def calcular_idf(docs):
  num_docs = len(docs)
  vocabulario, _ = calcular_frequencias(docs)
  idf = {}
  for palavra in vocabulario:
    num_docs_com_palavra = sum(1 for doc in docs if palavra in doc)
    idf[palavra] = np.log(num_docs / (1 + num_docs_com_palavra))
  return idf


def vetorizar_tfidf(docs):
  vocabulario, frequencias = calcular_frequencias(docs)
  idf = calcular_idf(docs)
  vocabulario = list(vocabulario)
  matriz_tfidf = []
  for freq in frequencias:
    vetor = [freq.get(palavra, 0) * idf.get(palavra, 0) for palavra in vocabulario]
    matriz_tfidf.append(vetor)
  return matriz_tfidf, vocabulario