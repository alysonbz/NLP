import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

corpus, labels = pd.read_csv("movie_reviews_clean.csv")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def load_dataset():
    df = pd.read_csv("movie_reviews_clean.csv")
    df = df.rename(columns={'review': 'text'})

    df["text"] = df["text"].apply(preprocess)
    return df

def evaluate(vectorizer_name, vectorizer, X_train, X_test, y_train, y_test):
    print(f"RESULTADOS USANDO {vectorizer_name}")

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)

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

    evaluate("TF-IDF", TfidfVectorizer(), X_train, X_test, y_train, y_test)
    evaluate("CountVectorizer", CountVectorizer(), X_train, X_test, y_train, y_test)