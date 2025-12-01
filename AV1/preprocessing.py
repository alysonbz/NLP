import re
import nltk
import spacy
import pandas as pd
from enelvo.normaliser import Normaliser
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

#nltk.download('rslp')
#nltk.download('stopwords')
nlp = spacy.load("pt_core_news_sm")

"""
Nesta primeira questão você deve implementar funções de manipulação do dataset realizar os pré-processmentos necessários, 
como stemming, lemmatização, remoção de carateris maiusculos, verificar stopwords, verificação de menções, de acordo as 
características do seu dataset. Em resume prepare o mesmo para aplicação de extração de atributos. A estrutura do código 
deve permitir que possam ser importadas as funções em outras questões.
"""

def carregar_dataset(caminho_arquivo, sep=","):
    return pd.read_csv(caminho_arquivo, encoding="utf-8", sep=sep)

def normalizar_texto(texto):
    norm = Normaliser(sanitize=True)
    return norm.normalise(texto)

def remover_links(texto):
    return re.sub(r"http\S+|www\S+", "", texto)

def remover_mencoes(texto):
    return re.sub(r"@\w+", "", texto)

def remover_numeros(texto):
    return re.sub(r"\d+", "", texto)

def tokenizar(texto):
    doc = nlp(texto)
    return [token.text for token in doc]

def lemmatizacao(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

def remover_stopwords(tokens):
    stopwords_pt = set(stopwords.words("portuguese"))
    return [t for t in tokens if t not in stopwords_pt]

def stemming(tokens):
    stemmer = RSLPStemmer()
    return [stemmer.stem(t) for t in tokens]