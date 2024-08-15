from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from preprocessing import preprocessamento, stemming
from vectorizer import count_vectorizer, tf_idf


# Carregar datasets
coment_pre_process = pd.read_csv("content_lemmatized_stemmed.csv")
coment_sem_tratamento = pd.read_csv("content_sem_trat.csv")

print('\nVerificar o balanceamento das classes:')
class_distribution = coment_pre_process['score'].value_counts()
print(class_distribution)

# Extrair labels
labels = coment_pre_process['score']

# a) Utilizando a vetorização por TF-IDF, compare os resultados de acerto do classificador
# com pré-processamento e sem pré-processamento. Mostre as taxas para os casos de forma organizada.

# Vetorização TF-IDF
tf_idf_vectors_preprocessed, _ = tf_idf(coment_pre_process['content_stemmed'])
tf_idf_vectors_raw, _ = tf_idf(coment_sem_tratamento['content'])

# Separação dos dados
X_train_preprocessed, X_test_preprocessed, y_train, y_test = train_test_split(tf_idf_vectors_preprocessed, labels, test_size=0.3)
X_train_raw, X_test_raw, _, _ = train_test_split(tf_idf_vectors_raw, labels, test_size=0.3)

# Classificador
model = MultinomialNB()
model.fit(X_train_preprocessed, y_train)
y_pred_preprocessed = model.predict(X_test_preprocessed)

model.fit(X_train_raw, y_train)
y_pred_raw = model.predict(X_test_raw)

print("TF-IDF com pré-processamento:")
print(classification_report(y_test, y_pred_preprocessed, zero_division=1))
print(confusion_matrix(y_test, y_pred_preprocessed))

print("TF-IDF sem pré-processamento:")
print(classification_report(y_test, y_pred_raw, zero_division=1))
print(confusion_matrix(y_test, y_pred_raw))

# b) Compare as vetorizações CountVectorizer x TF-IDF, usando com pré-processamento que
# você escolheu no item a)

# Comparação CountVectorizer x TF-IDF
count_vectors, _ = count_vectorizer(coment_pre_process['content_stemmed'])
X_train_count, X_test_count, _, _ = train_test_split(count_vectors, labels, test_size=0.3)

model.fit(X_train_count, y_train)
y_pred_count = model.predict(X_test_count)

print("CountVectorizer com pré-processamento:")
print(classification_report(y_test, y_pred_count, zero_division=1))
print(confusion_matrix(y_test, y_pred_count))


# C) Faça um variação de dois pré-procesamentos que compare lemmatização e steming,
# considerando a melhor forma de vetorização vista no item b)

# Variação entre lemmatização e stemming
coment_pre_process['content_stemmed'] = coment_pre_process['content'].apply(lambda x: preprocessamento(x, use_lemmatization=False))
coment_pre_process['content_lemmatized'] = coment_pre_process['content'].apply(lambda x: preprocessamento(x, use_lemmatization=True))

tf_idf_vectors_stemmed, _ = tf_idf(coment_pre_process['content_stemmed'])
tf_idf_vectors_lemmatized, _ = tf_idf(coment_pre_process['content_lemmatized'])

X_train_stemmed, X_test_stemmed, _, _ = train_test_split(tf_idf_vectors_stemmed, labels, test_size=0.3)
X_train_lemmatized, X_test_lemmatized, _, _ = train_test_split(tf_idf_vectors_lemmatized, labels, test_size=0.3)

model.fit(X_train_stemmed, y_train)
y_pred_stemmed = model.predict(X_test_stemmed)

print("TF-IDF com stemming:")
print(classification_report(y_test, y_pred_stemmed, zero_division=1))
print(confusion_matrix(y_test, y_pred_stemmed))

model.fit(X_train_lemmatized, y_train)
y_pred_lemmatized = model.predict(X_test_lemmatized)

print("TF-IDF com lemmatização:")
print(classification_report(y_test, y_pred_lemmatized, zero_division=1))
print(confusion_matrix(y_test, y_pred_lemmatized))
