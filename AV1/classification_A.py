import re
import nltk
import numpy as np
from scipy.sparse import csr_matrix
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from vectorizer import count_vectorizer, tfidf_vectorizer  # Importa as funções personalizadas
from preprocessing import preprocess_text  # Função de pré-processamento reutilizada

# Baixar pacotes adicionais do NLTK (caso necessário)
nltk.download('stopwords')
nltk.download('rslp')

# Carregar o dataset
ds = load_dataset("johnidouglas/twitter-sentiment-pt-BR-md-2-l")
df = ds['train'].to_pandas()
df_sample = df.sample(frac=0.2, random_state=42)  # Aumenta o tamanho da amostra

texts = df_sample['tweet_text'].values
labels = df_sample['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Sem pré-processamento
count_matrix_train, vocab = count_vectorizer(X_train)
count_matrix_test, _ = count_vectorizer(X_test)

tfidf_matrix_train, vocab = tfidf_vectorizer(X_train)
tfidf_matrix_test, _ = tfidf_vectorizer(X_test, vocab=vocab)

# Converter para sparse matrix
tfidf_matrix_train = csr_matrix(tfidf_matrix_train)
tfidf_matrix_test = csr_matrix(tfidf_matrix_test)

# Com pré-processamento usando a função do arquivo preprocessing
X_train_preprocessed = [preprocess_text(text) for text in X_train]
X_test_preprocessed = [preprocess_text(text) for text in X_test]

tfidf_matrix_train_pp, vocab_pp = tfidf_vectorizer(X_train_preprocessed)
tfidf_matrix_test_pp, _ = tfidf_vectorizer(X_test_preprocessed, vocab=vocab_pp)

# Converter para sparse matrix
tfidf_matrix_train_pp = csr_matrix(tfidf_matrix_train_pp)
tfidf_matrix_test_pp = csr_matrix(tfidf_matrix_test_pp)

clf = LogisticRegression(max_iter=1000)

# Sem pré-processamento
clf.fit(tfidf_matrix_train, y_train)
y_pred = clf.predict(tfidf_matrix_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"\nSem Pré-processamento:")
print(f"Acurácia: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nRelatório de Classificação (Sem Pré-processamento):\n")
print(classification_report(y_test, y_pred))

# Com pré-processamento
clf.fit(tfidf_matrix_train_pp, y_train)
y_pred_pp = clf.predict(tfidf_matrix_test_pp)

accuracy_pp = accuracy_score(y_test, y_pred_pp)
f1_pp = f1_score(y_test, y_pred_pp, average='weighted')
precision_pp = precision_score(y_test, y_pred_pp, average='weighted')
recall_pp = recall_score(y_test, y_pred_pp, average='weighted')

print(f"\nCom Pré-processamento:")
print(f"Acurácia: {accuracy_pp:.4f}")
print(f"F1-Score: {f1_pp:.4f}")
print(f"Precisão: {precision_pp:.4f}")
print(f"Recall: {recall_pp:.4f}")
print("\nRelatório de Classificação (Com Pré-processamento):\n")
print(classification_report(y_test, y_pred_pp))
