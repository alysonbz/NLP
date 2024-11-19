import numpy as np
from collections import Counter
import math

# Texto fornecido
texto = """Engraçados que nós seres humanos temos sempre afinidade com pessoas que querem o mesmo que nós, 
que pensa muitas das vezes igual, essas pessoas viram nossas amigas, romance e etc, mas se você reparar bem o seu inimigo 
é sempre aquele que se opõe a você, que nunca concorda com suas atitudes, e que nunca aceita a sua opinião."""

# Pré-processamento simples
def preprocessar_texto(texto):
    texto = texto.lower()  # Converter para letras minúsculas
    texto = texto.replace(",", "").replace(".", "").replace("!", "")  # Remover pontuação
    return texto.split()  # Dividir em palavras

# Calcular frequência dos termos (TF)
def calcular_tf(corpus):
    tf = []
    for doc in corpus:
        contador = Counter(doc)
        total_palavras = len(doc)
        tf.append({palavra: freq / total_palavras for palavra, freq in contador.items()})
    return tf

# Calcular IDF
def calcular_idf(corpus):
    num_documentos = len(corpus)
    todas_palavras = set(palavra for doc in corpus for palavra in doc)
    idf = {}
    for palavra in todas_palavras:
        num_docs_contem_palavra = sum(1 for doc in corpus if palavra in doc)
        idf[palavra] = math.log(num_documentos / (1 + num_docs_contem_palavra))
    return idf

# Calcular TF-IDF
def calcular_tfidf(tf, idf):
    tfidf = []
    for doc_tf in tf:
        tfidf.append({palavra: doc_tf[palavra] * idf[palavra] for palavra in doc_tf})
    return tfidf

# Preparar o corpus
documento = preprocessar_texto(texto)
corpus = [documento]  # Como só temos um documento, criamos uma lista com ele

# Cálculo de TF, IDF e TF-IDF
tf = calcular_tf(corpus)
idf = calcular_idf(corpus)
tfidf = calcular_tfidf(tf, idf)

# Resultado final
print("Matriz TF-IDF:")
for palavra, valor in tfidf[0].items():
    print(f"{palavra}: {valor:.4f}")
