import numpy as np


# Implementação manual de CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def count_vectorizer(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def tfidf_vectorizer(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer