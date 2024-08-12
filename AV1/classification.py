import numpy as np
import re
import nltk
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer, WordNetLemmatizer

# Baixar pacotes adicionais do NLTK
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('wordnet')

# Carregar o dataset
ds = load_dataset("johnidouglas/twitter-sentiment-pt-BR-md-2-l")

# Preparar os dados
texts = [example['tweet_text'] for example in ds['train']]
labels = [example['sentiment'] for example in ds['train']]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Função de pré-processamento com stemming
def preprocess_text_stemmed(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    stop_words = set(stopwords.words('portuguese'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    stemmer = RSLPStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Função de pré-processamento com lematização
def preprocess_text_lemmatized(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    stop_words = set(stopwords.words('portuguese'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Pré-processamento dos dados
X_train_stemmed = [preprocess_text_stemmed(text) for text in X_train]
X_test_stemmed = [preprocess_text_stemmed(text) for text in X_test]

X_train_lemmatized = [preprocess_text_lemmatized(text) for text in X_train]
X_test_lemmatized = [preprocess_text_lemmatized(text) for text in X_test]

# TF-IDF Sem Pré-processamento
vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)
y_pred_tfidf = classifier.predict(X_test_tfidf)

print("Resultados TF-IDF sem pré-processamento:")
print(classification_report(y_test, y_pred_tfidf))

# TF-IDF Com Stemming
vectorizer_tfidf_stemmed = TfidfVectorizer()
X_train_tfidf_stemmed = vectorizer_tfidf_stemmed.fit_transform(X_train_stemmed)
X_test_tfidf_stemmed = vectorizer_tfidf_stemmed.transform(X_test_stemmed)

classifier.fit(X_train_tfidf_stemmed, y_train)
y_pred_tfidf_stemmed = classifier.predict(X_test_tfidf_stemmed)

print("Resultados TF-IDF com stemming:")
print(classification_report(y_test, y_pred_tfidf_stemmed))

# TF-IDF Com Lematização
vectorizer_tfidf_lemmatized = TfidfVectorizer()
X_train_tfidf_lemmatized = vectorizer_tfidf_lemmatized.fit_transform(X_train_lemmatized)
X_test_tfidf_lemmatized = vectorizer_tfidf_lemmatized.transform(X_test_lemmatized)

classifier.fit(X_train_tfidf_lemmatized, y_train)
y_pred_tfidf_lemmatized = classifier.predict(X_test_tfidf_lemmatized)

print("Resultados TF-IDF com lematização:")
print(classification_report(y_test, y_pred_tfidf_lemmatized))

# CountVectorizer Com Stemming
count_vectorizer_stemmed = CountVectorizer()
X_train_count_stemmed = count_vectorizer_stemmed.fit_transform(X_train_stemmed)
X_test_count_stemmed = count_vectorizer_stemmed.transform(X_test_stemmed)

classifier.fit(X_train_count_stemmed, y_train)
y_pred_count_stemmed = classifier.predict(X_test_count_stemmed)

print("Resultados CountVectorizer com stemming:")
print(classification_report(y_test, y_pred_count_stemmed))

# CountVectorizer Com Lematização
count_vectorizer_lemmatized = CountVectorizer()
X_train_count_lemmatized = count_vectorizer_lemmatized.fit_transform(X_train_lemmatized)
X_test_count_lemmatized = count_vectorizer_lemmatized.transform(X_test_lemmatized)

classifier.fit(X_train_count_lemmatized, y_train)
y_pred_count_lemmatized = classifier.predict(X_test_count_lemmatized)

print("Resultados CountVectorizer com lematização:")
print(classification_report(y_test, y_pred_count_lemmatized))
