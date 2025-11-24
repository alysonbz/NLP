import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import load_movie_review_clean_dataset

corpus, labels = load_movie_review_clean_dataset()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def load_dataset():
    df = load_movie_review_clean_dataset() 
    df = df.rename(columns={'review': 'text'}) 

    df["text"] = df["text"].apply(preprocess)
    return df

def evaluate(vectorizer_name, vectorizer, X_train, X_test, y_train, y_test):
    print(f"RESULTADOS USANDO {vectorizer_name}")

    # Vetorização
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Modelo
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)

    # Métricas
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, preds))
    print("\n")


if __name__ == "__main__":
    df = load_dataset()

    X = df["text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Comparação dos vetorizadores
    evaluate("TF-IDF", TfidfVectorizer(), X_train, X_test, y_train, y_test)
    evaluate("CountVectorizer", CountVectorizer(), X_train, X_test, y_train, y_test)


"""
COMPARAÇÃO:
Os resultados mostram que o modelo Naive Bayes tem desempenho muito inferior quando utiliza TF-IDF,
pois embora identifique quase todos os textos da classe negativa, apresenta recall extremamente baixo para a classe positiva.
Já com o CountVectorizer, que trabalha com contagens brutas — algo muito mais adequado ao funcionamento do Naive Bayes — 
o modelo consegue reconhecer bem ambas as classes, alcançando resultados muito mais equilibrados.
"""