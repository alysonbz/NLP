import numpy as np
from scipy.sparse import csr_matrix
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from vectorizer import tfidf_vectorizer  # Importando a função TF-IDF
from preprocessing import preprocess_text  # Importando o pré-processamento

# Carregar o dataset
ds = load_dataset("johnidouglas/twitter-sentiment-pt-BR-md-2-l")
df = ds['train'].to_pandas()
df_sample = df.sample(frac=0.2, random_state=42)  # Aumenta o tamanho da amostra

texts = df_sample['tweet_text'].values
labels = df_sample['sentiment'].values

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Aplicando o pré-processamento (stem e lemma)
X_train_stemmed = [preprocess_text(text, apply_stem=True) for text in X_train]
X_test_stemmed = [preprocess_text(text, apply_stem=True) for text in X_test]

X_train_lemmatized = [preprocess_text(text, apply_lemmatize=True) for text in X_train]
X_test_lemmatized = [preprocess_text(text, apply_lemmatize=True) for text in X_test]

# Vetorização e Treinamento com TF-IDF e Stemming
tfidf_matrix_train_stem, vocab_tfidf_stem = tfidf_vectorizer(X_train_stemmed)
tfidf_matrix_test_stem, _ = tfidf_vectorizer(X_test_stemmed, vocab=vocab_tfidf_stem)

tfidf_matrix_train_stem = csr_matrix(tfidf_matrix_train_stem)
tfidf_matrix_test_stem = csr_matrix(tfidf_matrix_test_stem)

clf = LogisticRegression(max_iter=1000)
clf.fit(tfidf_matrix_train_stem, y_train)
y_pred_tfidf_stem = clf.predict(tfidf_matrix_test_stem)

accuracy_tfidf_stem = accuracy_score(y_test, y_pred_tfidf_stem)
f1_tfidf_stem = f1_score(y_test, y_pred_tfidf_stem, average='weighted')
precision_tfidf_stem = precision_score(y_test, y_pred_tfidf_stem, average='weighted')
recall_tfidf_stem = recall_score(y_test, y_pred_tfidf_stem, average='weighted')

print(f"\nTF-IDF com Stemming:")
print(f"Acurácia: {accuracy_tfidf_stem:.4f}")
print(f"F1-Score: {f1_tfidf_stem:.4f}")
print(f"Precisão: {precision_tfidf_stem:.4f}")
print(f"Recall: {recall_tfidf_stem:.4f}")
print("\nRelatório de Classificação (TF-IDF com Stemming):\n")
print(classification_report(y_test, y_pred_tfidf_stem))

# Vetorização e Treinamento com TF-IDF e Lematização
tfidf_matrix_train_lemma, vocab_tfidf_lemma = tfidf_vectorizer(X_train_lemmatized)
tfidf_matrix_test_lemma, _ = tfidf_vectorizer(X_test_lemmatized, vocab=vocab_tfidf_lemma)

tfidf_matrix_train_lemma = csr_matrix(tfidf_matrix_train_lemma)
tfidf_matrix_test_lemma = csr_matrix(tfidf_matrix_test_lemma)

clf.fit(tfidf_matrix_train_lemma, y_train)
y_pred_tfidf_lemma = clf.predict(tfidf_matrix_test_lemma)

accuracy_tfidf_lemma = accuracy_score(y_test, y_pred_tfidf_lemma)
f1_tfidf_lemma = f1_score(y_test, y_pred_tfidf_lemma, average='weighted')
precision_tfidf_lemma = precision_score(y_test, y_pred_tfidf_lemma, average='weighted')
recall_tfidf_lemma = recall_score(y_test, y_pred_tfidf_lemma, average='weighted')

print(f"\nTF-IDF com Lematização:")
print(f"Acurácia: {accuracy_tfidf_lemma:.4f}")
print(f"F1-Score: {f1_tfidf_lemma:.4f}")
print(f"Precisão: {precision_tfidf_lemma:.4f}")
print(f"Recall: {recall_tfidf_lemma:.4f}")
print("\nRelatório de Classificação (TF-IDF com Lematização):\n")
print(classification_report(y_test, y_pred_tfidf_lemma))
