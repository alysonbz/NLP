import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline


# Função para pré-processar o texto
def preprocess_text(text):
    text = text.lower()  # Converte para minúsculas
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove pontuação
    text = re.sub(r"\d+", "", text)  # Remove números
    text = re.sub(r"\s+", " ", text).strip()  # Remove espaços extras
    return text


# Função para avaliar o modelo
def evaluate_model(X, y, vectorizer, classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    # Avaliação do modelo
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return report, conf_matrix


# Função para plotar a matriz de confusão
def plot_confusion_matrix(conf_matrix, labels, title="Matriz de Confusão"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()


# Função principal s
def run_evaluation(corpus, classifiers, vectorizers):
    corpus["clean_review"] = corpus["review"].apply(preprocess_text)  # Pré-processamento
    results = {}

    for name, classifier in classifiers.items():
        print(f"\nAvaliação com o classificador: {name}")
        results[name] = {}
        for vec_name, vectorizer in vectorizers.items():
            print(f"\n - Vetorização: {vec_name}")
            report, conf_matrix = evaluate_model(corpus["clean_review"], corpus["sentiment"], vectorizer, classifier)
            results[name][vec_name] = (report, conf_matrix)
            print("Classification Report:")
            print(pd.DataFrame(report).transpose())
            plot_confusion_matrix(conf_matrix, labels=["Negativo", "Positivo"],
                                  title=f"Matriz de Confusão - {name} - {vec_name}")

    return results


if __name__ == "__main__":
    # Carregar o dataset
    from src.utils import load_movie_review_clean_dataset

    corpus = load_movie_review_clean_dataset()

    # Definir classificadores
    classifiers = {
        "MultinomialNB": MultinomialNB(),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    # Definir vetorizadores
    vectorizers = {
        "TF-IDF": TfidfVectorizer(),
        "CountVectorizer": CountVectorizer()
    }

    # Executar avaliação
    results = run_evaluation(corpus, classifiers, vectorizers)

    # Resultados finais
    print("\nResultados Finais:")
    for classifier_name, vector_results in results.items():
        for vec_name, (report, conf_matrix) in vector_results.items():
            print(f"\nClassificador: {classifier_name}, Vetorizador: {vec_name}")
            print(pd.DataFrame(report).transpose())
