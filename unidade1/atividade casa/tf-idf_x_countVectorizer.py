import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import load_movie_review_clean_dataset


def preprocess(texts):
    """
    Preprocess texts by tokenizing and converting to lowercase.
    """
    # Adicione outros passos de pré-processamento se necessário
    return texts


def main():
    # Carregar o dataset limpo usando a função fornecida
    corpus = load_movie_review_clean_dataset()

    # Verificar e imprimir o tipo e o formato dos dados
    print(f"Tipo do corpus: {type(corpus)}")
    print(f"Primeiros 5 itens do corpus:\n{corpus.head()}")

    # Separar os textos e rótulos do DataFrame
    texts = corpus['review'].tolist()
    labels = corpus['sentiment'].tolist()

    # Pré-processar os textos
    texts = preprocess(texts)

    # Dividir o dataset em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

    # Vetorização com TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Vetorização com CountVectorizer
    count_vectorizer = CountVectorizer()
    X_train_count = count_vectorizer.fit_transform(X_train)
    X_test_count = count_vectorizer.transform(X_test)

    # Classificador
    classifier = LogisticRegression(max_iter=1000)

    # Treinamento e avaliação com TF-IDF
    classifier.fit(X_train_tfidf, y_train)
    y_pred_tfidf = classifier.predict(X_test_tfidf)
    print("Resultados com TF-IDF:")
    print(classification_report(y_test, y_pred_tfidf))
    print("Matriz de Confusão com TF-IDF:")
    print(confusion_matrix(y_test, y_pred_tfidf))

    # Treinamento e avaliação com CountVectorizer
    classifier.fit(X_train_count, y_train)
    y_pred_count = classifier.predict(X_test_count)
    print("Resultados com CountVectorizer:")
    print(classification_report(y_test, y_pred_count))
    print("Matriz de Confusão com CountVectorizer:")
    print(confusion_matrix(y_test, y_pred_count))


if __name__ == "__main__":
    main()
