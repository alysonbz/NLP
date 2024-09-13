import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import load_movie_review_clean_dataset

# Carregar o dataset
corpus = load_movie_review_clean_dataset()
X = corpus['review']
y = corpus['sentiment']


# Função para imprimir a matriz de confusão e classification report
def evaluate_model(y_true, y_pred, title):
    print(f"\n{title}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {title}')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.show()


# Pré-processamento e divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vetorização com CountVectorizer
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# Treinamento com MultinomialNB e CountVectorizer
clf_count = MultinomialNB()
clf_count.fit(X_train_count, y_train)
y_pred_count = clf_count.predict(X_test_count)

# Avaliação CountVectorizer
evaluate_model(y_test, y_pred_count, 'CountVectorizer')

# Vetorização com TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Treinamento com MultinomialNB e TF-IDF
clf_tfidf = MultinomialNB()
clf_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)

# Avaliação TF-IDF
evaluate_model(y_test, y_pred_tfidf, 'TF-IDF')

# Comparação entre CountVectorizer e TF-IDF
print("Resultados comparados:")
print(f"CountVectorizer -> Acurácia: {np.mean(y_pred_count == y_test):.4f}")
print(f"TF-IDF -> Acurácia: {np.mean(y_pred_tfidf == y_test):.4f}")
