"""
Nesta primeira questão você deve implementar funções de manipulação do dataset realizar os pré-processmentos necessários, 
como stemming, lemmatização, remoção de carateris maiusculos, verificar stopwords, verificação de menções, de acordo as 
características do seu dataset. Em resume prepare o mesmo para aplicação de extração de atributos. A estrutura do código 
deve permitir que possam ser importadas as funções em outras questões.
"""

import re
import spacy
import pandas as pd
from enelvo.normaliser import Normaliser
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# --- Carregamento único de recursos pesados ---
nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"]) 
normaliser = Normaliser(sanitize=True)
stopwords_pt = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()

# Regex pré-compiladas
regex_links = re.compile(r"http\S+|www\S+")
regex_mencoes = re.compile(r"@\w+")
regex_numeros = re.compile(r"\d+")



def carregar_dataset(caminho_arquivo, sep=","):
    """Carrega dataset no formato CSV."""
    return pd.read_csv(caminho_arquivo, encoding="utf-8", sep=sep)


def normalizar_texto(texto):
    """Normalização com Enelvo."""
    return normaliser.normalise(texto)


def remover_links(texto):
    return regex_links.sub("", texto)


def remover_mencoes(texto):
    return regex_mencoes.sub("", texto)


def remover_numeros(texto):
    return regex_numeros.sub("", texto)


def limpar_texto(texto):
    """Pipeline básico de limpeza antes da tokenização."""
    texto = texto.lower()
    texto = remover_links(texto)
    texto = remover_mencoes(texto)
    texto = remover_numeros(texto)
    texto = normalizar_texto(texto)
    return texto.strip()


def tokenizar(texto):
    """Tokenização rápida com spaCy."""
    return [token.text for token in nlp(texto)]


def lemmatizacao(tokens):
    """Lemmatização usando spaCy."""
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]


def remover_stopwords(tokens):
    return [t for t in tokens if t not in stopwords_pt and len(t) > 1]


def stemming(tokens):
    return [stemmer.stem(t) for t in tokens]


def preprocessar(texto, aplicar_lemma=True, aplicar_stem=False):
    """
    Pipeline completo de pré-processamento.
    """
    texto = limpar_texto(texto)
    tokens = tokenizar(texto)

    if aplicar_lemma:
        tokens = lemmatizacao(tokens)

    tokens = remover_stopwords(tokens)

    if aplicar_stem:
        tokens = stemming(tokens)

    return tokens