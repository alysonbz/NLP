from src.utils import load_movie_review_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



def preprocessar(texto):
    return texto.lower()



def plotar_matriz_confusao(y_true, y_pred, titulo):
    matriz = confusion_matrix(y_true, y_pred)
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues")
    plt.title(titulo)
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.show()



corpus = load_movie_review_clean_dataset()
X = corpus['review']
y = corpus['sentiment']

X = X.apply(preprocessar)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



vectorizer_bow = CountVectorizer(stop_words='english')
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

classificador_bow = MultinomialNB()
classificador_bow.fit(X_train_bow, y_train)

y_pred_bow = classificador_bow.predict(X_test_bow)

print('\nResultados (CountVectorizer):\n')
print(classification_report(y_test, y_pred_bow))

plotar_matriz_confusao(y_test, y_pred_bow, titulo='Matriz de confusão (bag of words)')

# Modelo com o TF-IDF:
vectorizer_tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

classificador_tfidf = MultinomialNB()
classificador_tfidf.fit(X_train_tfidf, y_train)

y_pred_tfidf = classificador_tfidf.predict(X_test_tfidf)

print('\nResultados (TF-IDF):\n')
print(classification_report(y_test, y_pred_tfidf))

plotar_matriz_confusao(y_test, y_pred_tfidf, 'Matriz de confusão - TF-IDF')