# preprocessing.py
# Funções de pré-processamento do dataset

import re
import ast
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Carregar modelos
nlp = spacy.load("pt_core_news_sm")
stopwords_pt = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()


# Limpeza básica
def normalize_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()                              # minúsculas
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)                 # remove menções (@usuario)
    text = re.sub(r"[^a-záéíóúâêôãõç ]", " ", text)  # remove pontuação e números
    text = re.sub(r"\s+", " ", text).strip()         # remove múltiplos espaços
    return text


# Remoção de stopwords
def remove_stopwords(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_pt]
    return " ".join(tokens)


# Stemming
def apply_stemming(text):
    return " ".join(stemmer.stem(t) for t in text.split())


# Lemmatização
def apply_lemmatization(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)


# Limpeza da coluna keywords
def clean_keywords(kw_string):
    # converte string "['a','b']" para lista real
    try:
        keywords = ast.literal_eval(kw_string)
        if isinstance(keywords, list):
            return [normalize_text(k) for k in keywords]
    except:
        return []
    return []


# Pipeline principal
def preprocess_text(text, do_stem=False, do_lemma=True):
    text = normalize_text(text)
    text = remove_stopwords(text)

    if do_stem:
        text = apply_stemming(text)

    if do_lemma:
        text = apply_lemmatization(text)

    return text


# Processar o dataset inteiro
def preprocess_dataset(df):
    df = df.copy()

    # Processar títulos
    df["headline_clean"] = df["headlinePortuguese"].apply(preprocess_text)

    # Processar keywords
    df["keywords_clean"] = df["keywords"].apply(clean_keywords)

    return df

# from preprocessing import preprocess_dataset

# df = pd.read_csv("brazilian_headlines_sentiments.csv")

# df_clean = preprocess_dataset(df)

# df_clean.head()