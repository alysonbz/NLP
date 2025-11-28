import string
import kagglehub
import pandas as pd
from pathlib import Path
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
import nltk
import re
import spacy
nlp = spacy.load("pt_core_news_sm")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('rslp')

"""Nesta primeira questão você deve implementar funções de manipulação do dataset
realizar os pré-processmentos necessários, como stemming, lemmatização,
remoção de carateris maiusculos, verificar stopwords, verificação de menções,
de acordo as características do seu dataset. Em resume prepare o mesmo para
aplicação de extração de atributos. A estrutura do código deve permitir
que possam ser importadas as funções em outras questões."""

def ler_datasets():
    """Carregar os datasets diretamente do kaggle"""
    path = kagglehub.dataset_download("moesiof/portuguese-narrative-essays")
    path = Path(path)

    print("Path to dataset files:", path)

    train = pd.read_csv(path / "train.csv")
    test = pd.read_csv(path / "test.csv")
    validation = pd.read_csv(path / "validation.csv")

    return train, test, validation

def stemming(text):
    stemmer = RSLPStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def lemmatization(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)

def remover_uppercase(text):
    return text.lower()

def remover_stopwords(text):
    stop_words = set(stopwords.words('portuguese'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remover_simbolos(texto):
    padrao = r'\[P\]|\[\s?P\]|\[P\}|\[p\]|\{p\}|' \
             r'\[S\]|\[s\]|\[R\]|\[r\]|\[X\]|\[x\]|\[X~\]|\{x\}|' \
             r'\[T\]|\[t\]|\{t\}|\[\?\]|\{\?\}|\[\?}|\{\?\]|' \
             r'\[LC\]|\[LT\]|\[lt\]'

    # remove tudo que bater com o padrão
    texto_limpo = re.sub(padrao, '', texto)

    # colapsa múltiplos espaços deixados para trás
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    return texto_limpo

def normalizar_espacos(text):
    return re.sub(r'\s+', ' ', text).strip()

def remover_pontuacao(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remover_numeros(text):
    return re.sub(r"\d+", "", text)