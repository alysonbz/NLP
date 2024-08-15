import numpy as np
from scipy.sparse import csr_matrix
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from vectorizer import count_vectorizer, tfidf_vectorizer  # Importando as funções personalizadas
from preprocessing import preprocess_text  # Importando o mesmo pré-processamento de classification_A

# Carregar o dataset
ds = load_dataset("johnidouglas/twitter-sentiment-pt-BR-md-2-l")
df = ds['train'].to_pandas()
df_sample = df.sample(frac=0.2, random_state=42)  # Aumenta o tamanho da amostra

texts = df_sample['tweet_text'].values
labels = df_sample['sentiment'].values

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Aplicando o pré-processamento (o mesmo de classification_A)
X_train_preprocessed = [preprocess_text(text) for text in X_train]
X_test_preprocessed = [preprocess_text(text) for text in X_test]

# CountVectorizer
count_matrix_train, vocab_count = count_vectorizer(X_train_preprocessed)
count_matrix_test, _ = count_vectorizer(X_test_preprocessed, vocab=vocab_count)  # Usar o mesmo vocabulário

# Converter para sparse matrix
count_matrix_train = csr_matrix(count_matrix_train)
count_matrix_test = csr_matrix(count_matrix_test)

# TF-IDF
tfidf_matrix_train_pp, vocab_tfidf = tfidf_vectorizer(X_train_preprocessed, vocab=vocab_count)
tfidf_matrix_test_pp, _ = tfidf_vectorizer(X_test_preprocessed, vocab=vocab_tfidf)  # Usar o mesmo vocabulário

# Converter para sparse matrix
tfidf_matrix_train_pp = csr_matrix(tfidf_matrix_train_pp)
tfidf_matrix_test_pp = csr_matrix(tfidf_matrix_test_pp)

# Modelo de Regressão Logística
clf = LogisticRegression(max_iter=1000)

# CountVectorizer
clf.fit(count_matrix_train, y_train)
y_pred_count = clf.predict(count_matrix_test)

accuracy_count = accuracy_score(y_test, y_pred_count)
f1_count = f1_score(y_test, y_pred_count, average='weighted')
precision_count = precision_score(y_test, y_pred_count, average='weighted')
recall_count = recall_score(y_test, y_pred_count, average='weighted')

print(f"\nCountVectorizer:")
print(f"Acurácia: {accuracy_count:.4f}")
print(f"F1-Score: {f1_count:.4f}")
print(f"Precisão: {precision_count:.4f}")
print(f"Recall: {recall_count:.4f}")
print("\nRelatório de Classificação (CountVectorizer):\n")
print(classification_report(y_test, y_pred_count))

# TF-IDF
clf.fit(tfidf_matrix_train_pp, y_train)
y_pred_tfidf = clf.predict(tfidf_matrix_test_pp)

accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
f1_tfidf = f1_score(y_test, y_pred_tfidf, average='weighted')
precision_tfidf = precision_score(y_test, y_pred_tfidf, average='weighted')
recall_tfidf = recall_score(y_test, y_pred_tfidf, average='weighted')

print(f"\nTF-IDF:")
print(f"Acurácia: {accuracy_tfidf:.4f}")
print(f"F1-Score: {f1_tfidf:.4f}")
print(f"Precisão: {precision_tfidf:.4f}")
print(f"Recall: {recall_tfidf:.4f}")
print("\nRelatório de Classificação (TF-IDF):\n")
print(classification_report(y_test, y_pred_tfidf))
