from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.utils import load_movie_review_clean_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Carregar o dataset
corpus, labels = load_movie_review_clean_dataset()

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)

# Vetorizar os dados com CountVectorizer
count_vect = CountVectorizer()
X_train_count = count_vect.fit_transform(X_train)
X_test_count = count_vect.transform(X_test)

# Vetorizar os dados com TF-IDF
tfidf_vect = TfidfVectorizer()
X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

# Função para treinar e avaliar o classificador
def treinar_avaliar_modelo(X_train, X_test, y_train, y_test, metodo):
    # Treinar o modelo
    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)

    # Predições
    y_pred = modelo.predict(X_test)

    # Avaliação
    print(f"\n--- Avaliação com {metodo} ---")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Matriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot da matriz de confusão
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    plt.title(f"Matriz de Confusão - {metodo}")
    plt.xlabel("Predição")
    plt.ylabel("Verdadeiro")
    plt.show()

    # Retornar acurácia para comparação
    return accuracy_score(y_test, y_pred)

# Avaliar com CountVectorizer
acc_count = treinar_avaliar_modelo(X_train_count, X_test_count, y_train, y_test, "CountVectorizer")

# Avaliar com TF-IDF
acc_tfidf = treinar_avaliar_modelo(X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF")

# Comparação das acurácias
print(f"\nAcurácia com CountVectorizer: {acc_count:.4f}")
print(f"Acurácia com TF-IDF: {acc_tfidf:.4f}")
