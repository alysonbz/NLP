import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer, PorterStemmer

from AV1.preprocessing import *
from AV1.vectorizer import *

dataset = load_portuguese_hate_speech()
dataset['cleaned_text'] = dataset['text'].apply(preprocess_text)

# Vamos considerar a coluna 'hatespeech_comb' como a label que desejamos prever
X_raw = dataset['text']
X_cleaned = dataset['cleaned_text']
y = dataset['hatespeech_comb']

# Vetorização TF-IDF no texto original
X_raw_tfidf = manual_tf_idf(X_raw)
#print(X_raw_tfidf)
#print(len(X_raw_tfidf))


# Vetorização TF-IDF no texto pré-processado
X_cleaned_tfidf = manual_tf_idf(X_cleaned)
#print(X_cleaned_tfidf)
#print(len(X_cleaned_tfidf))


# Dividindo os dados em treino e teste
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw_tfidf, y, test_size=0.2, random_state=42)
X_train_cleaned, X_test_cleaned, _, _ = train_test_split(X_cleaned_tfidf, y, test_size=0.2, random_state=42)


# Treinando o classificador com o texto original usando Logistic Regression
clf_raw_lr = LogisticRegression(max_iter=1000)
clf_raw_lr.fit(X_train_raw, y_train)
y_pred_raw_lr = clf_raw_lr.predict(X_test_raw)

# Treinando o classificador com o texto pré-processado usando Logistic Regression
clf_cleaned_lr = LogisticRegression(max_iter=1000)
clf_cleaned_lr.fit(X_train_cleaned, y_train)
y_pred_cleaned_lr = clf_cleaned_lr.predict(X_test_cleaned)

# Avaliando a precisão
accuracy_raw_lr = accuracy_score(y_test, y_pred_raw_lr)
accuracy_cleaned_lr = accuracy_score(y_test, y_pred_cleaned_lr)

print("------------------ USANDO REGRESÃO LOGISTICA -----------------------")
print(f"Acurácia com texto original (sem pré-processamento) usando Logistic Regression: {accuracy_raw_lr:.4f}")
print(f"Acurácia com texto pré-processado usando Logistic Regression: {accuracy_cleaned_lr:.4f}")
print("\n")

# Vetorização CountVectorizer no texto pré-processado
X_cleaned_count = manual_count_vectorizer(X_cleaned)

# Dividindo os dados em treino e teste para CountVectorizer
X_train_cleaned_count, X_test_cleaned_count, _, _ = train_test_split(X_cleaned_count, y, test_size=0.2, random_state=42)

# Treinando o classificador com o texto pré-processado usando CountVectorizer
clf_cleaned_count_lr = LogisticRegression(max_iter=1000)
clf_cleaned_count_lr.fit(X_train_cleaned_count, y_train)
y_pred_cleaned_count_lr = clf_cleaned_count_lr.predict(X_test_cleaned_count)

# Avaliando a precisão
accuracy_cleaned_count_lr = accuracy_score(y_test, y_pred_cleaned_count_lr)
print("------------------ COMPARANDO TF-IDF E COUNT VECTORIZER -----------------------")
print(f"Acurácia com texto pré-processado usando TF-IDF: {accuracy_cleaned_lr:.4f}")
print(f"Acurácia com texto pré-processado usando CountVectorizer: {accuracy_cleaned_count_lr:.4f}")
print("\n")


# Inicializando o lemmatizer e o stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Função para lematizar o texto
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Função para stemming do texto
def stem_text(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Aplicando lemmatization e stemming
X_lemmatized = X_cleaned.apply(lemmatize_text)
X_stemmed = X_cleaned.apply(stem_text)

# Vetorizando os textos lemmatizados
X_lemmatized_tfidf = manual_tf_idf(X_lemmatized)
X_stemmed_tfidf = manual_tf_idf(X_stemmed)

# Dividindo os dados em treino e teste
X_train_lemmatized, X_test_lemmatized, _, _ = train_test_split(X_lemmatized_tfidf, y, test_size=0.2, random_state=42)
X_train_stemmed, X_test_stemmed, _, _ = train_test_split(X_stemmed_tfidf, y, test_size=0.2, random_state=42)

# Treinando com o textopip lemmatizado
clf_lemmatized_lr = LogisticRegression(max_iter=1000)
clf_lemmatized_lr.fit(X_train_lemmatized, y_train)
y_pred_lemmatized_lr = clf_lemmatized_lr.predict(X_test_lemmatized)

# Treinando com o texto stemming
clf_stemmed_lr = LogisticRegression(max_iter=1000)
clf_stemmed_lr.fit(X_train_stemmed, y_train)
y_pred_stemmed_lr = clf_stemmed_lr.predict(X_test_stemmed)

# Avaliando a precisão
accuracy_lemmatized_lr = accuracy_score(y_test, y_pred_lemmatized_lr)
accuracy_stemmed_lr = accuracy_score(y_test, y_pred_stemmed_lr)
print("------------------ USANDO LEMMA E STEMMING -----------------------")
print(f"Acurácia com lemmatização: {accuracy_lemmatized_lr:.4f}")
print(f"Acurácia com stemming: {accuracy_stemmed_lr:.4f}")
print("\n")
