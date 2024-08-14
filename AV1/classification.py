import random
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import wordnet
import numpy as np
from vectorizer import count_vectorizer, tfidf_vectorizer
from preprocessing import preprocess_text, synonym_replacement,random_insertion, preprocess_with_stemming, preprocess_with_lemmatization

#Carregando Dados
ds = load_dataset("nilc-nlp/assin", "ptbr")
df = pd.DataFrame(ds['train'])


X_raw = df['premise'] + " " + df['hypothesis']  # Combinando 'premise' e 'hypothesis' como entradas
y = df['entailment_judgment']  # Usando a coluna de labels


X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)


X_augmented_raw = pd.concat([
    X_train_raw.apply(lambda x: synonym_replacement(x, n=2)),
    X_train_raw.apply(lambda x: random_insertion(x, n=2))
])


X_train_combined_raw = pd.concat([X_train_raw, X_augmented_raw], axis=0)
y_train_combined_raw = pd.concat([y_train] * 3, axis=0)


vocab_tfidf_raw, X_train_tfidf_raw = tfidf_vectorizer(X_train_combined_raw)


X_test_tfidf_raw = np.zeros((len(X_test_raw), len(vocab_tfidf_raw)))
for i, document in enumerate(X_test_raw):
    for word in document.split():
        if word in vocab_tfidf_raw:
            X_test_tfidf_raw[i, vocab_tfidf_raw[word]] += 1


X_test_tfidf_raw = X_test_tfidf_raw * np.log((len(X_train_combined_raw) + 1) / (np.sum(X_test_tfidf_raw > 0, axis=0) + 1)) + 1


clf_raw = LogisticRegression()
clf_raw.fit(X_train_tfidf_raw, y_train_combined_raw)


y_pred_raw = clf_raw.predict(X_test_tfidf_raw)
accuracy_raw = accuracy_score(y_test, y_pred_raw)


print("Classification Report Sem Pré-processamento:")
print(classification_report(y_test, y_pred_raw))


X_train_preprocessed = X_train_raw.apply(preprocess_text)
X_test_preprocessed = X_test_raw.apply(preprocess_text)


X_augmented_preprocessed = pd.concat([
    X_train_preprocessed.apply(lambda x: synonym_replacement(x, n=2)),
    X_train_preprocessed.apply(lambda x: random_insertion(x, n=2))
])


X_train_combined_preprocessed = pd.concat([X_train_preprocessed, X_augmented_preprocessed], axis=0)
y_train_combined_preprocessed = pd.concat([y_train] * 3, axis=0)


vocab_tfidf_preprocessed, X_train_tfidf_preprocessed = tfidf_vectorizer(X_train_combined_preprocessed)


X_test_tfidf_preprocessed = np.zeros((len(X_test_preprocessed), len(vocab_tfidf_preprocessed)))
for i, document in enumerate(X_test_preprocessed):
    for word in document.split():
        if word in vocab_tfidf_preprocessed:
            X_test_tfidf_preprocessed[i, vocab_tfidf_preprocessed[word]] += 1


X_test_tfidf_preprocessed = X_test_tfidf_preprocessed * np.log((len(X_train_combined_preprocessed) + 1) / (np.sum(X_test_tfidf_preprocessed > 0, axis=0) + 1)) + 1


clf_preprocessed = LogisticRegression()
clf_preprocessed.fit(X_train_tfidf_preprocessed, y_train_combined_preprocessed)


y_pred_preprocessed = clf_preprocessed.predict(X_test_tfidf_preprocessed)
accuracy_preprocessed = accuracy_score(y_test, y_pred_preprocessed)

print("Classification Report Com Pré-processamento:")
print(classification_report(y_test, y_pred_preprocessed))


###item b)###
"""Compare as vetorizações CountVectorizer x TF-IDF, usando com pré-processamento que você escolheu no item a)"""

# Extraindo as colunas de interesse
X_raw = df['premise'] + " " + df['hypothesis']  # Combinando 'premise' e 'hypothesis' como entradas
y = df['entailment_judgment']  # Usando a coluna de labels

# Dividindo os dados em conjuntos de treino e teste
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

# Aplicando o pré-processamento
X_train_preprocessed = X_train_raw.apply(preprocess_text)
X_test_preprocessed = X_test_raw.apply(preprocess_text)

# Aplicando Data Augmentation: Synonym Replacement e Random Insertion
X_augmented_preprocessed = pd.concat([
    X_train_preprocessed.apply(lambda x: synonym_replacement(x, n=2)),
    X_train_preprocessed.apply(lambda x: random_insertion(x, n=2))
])

# Combinando o dataset original com o aumentado
X_train_combined_preprocessed = pd.concat([X_train_preprocessed, X_augmented_preprocessed], axis=0)
y_train_combined_preprocessed = pd.concat([y_train] * 3, axis=0)

# Vetorização com CountVectorizer implementado manualmente
vocab_count_preprocessed, X_train_count_preprocessed = count_vectorizer(X_train_combined_preprocessed)

# Vetorização no conjunto de teste usando o vocabulário do conjunto de treino
X_test_count_preprocessed = np.zeros((len(X_test_preprocessed), len(vocab_count_preprocessed)))
for i, document in enumerate(X_test_preprocessed):
    for word in document.split():
        if word in vocab_count_preprocessed:
            X_test_count_preprocessed[i, vocab_count_preprocessed[word]] += 1

# Treinando o classificador com CountVectorizer implementado manualmente
clf_count = LogisticRegression()
clf_count.fit(X_train_count_preprocessed, y_train_combined_preprocessed)

# Prevendo e calculando a acurácia com CountVectorizer implementado manualmente
y_pred_count = clf_count.predict(X_test_count_preprocessed)
accuracy_count = accuracy_score(y_test, y_pred_count)

# Exibindo o classification report para o caso com CountVectorizer implementado manualmente
print("Classification Report com CountVectorizer:")
print(classification_report(y_test, y_pred_count))

# Vetorização com TF-IDF
vocab_tfidf_preprocessed, X_train_tfidf_preprocessed = tfidf_vectorizer(X_train_combined_preprocessed)

# Vetorização no conjunto de teste usando o vocabulário do conjunto de treino
X_test_tfidf_preprocessed = np.zeros((len(X_test_preprocessed), len(vocab_tfidf_preprocessed)))
for i, document in enumerate(X_test_preprocessed):
    for word in document.split():
        if word in vocab_tfidf_preprocessed:
            X_test_tfidf_preprocessed[i, vocab_tfidf_preprocessed[word]] += 1

# Normalizando as frequências de TF-IDF
X_test_tfidf_preprocessed = X_test_tfidf_preprocessed * np.log((len(X_train_combined_preprocessed) + 1) / (np.sum(X_test_tfidf_preprocessed > 0, axis=0) + 1)) + 1

# Treinando o classificador com TF-IDF
clf_tfidf = LogisticRegression()
clf_tfidf.fit(X_train_tfidf_preprocessed, y_train_combined_preprocessed)

# Prevendo e calculando a acurácia com TF-IDF
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf_preprocessed)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)

# Exibindo o classification report para o caso com TF-IDF
print("Classification Report com TF-IDF ")
print(classification_report(y_test, y_pred_tfidf))



###Item C)###
"""Faça um variação de dois pré-procesamentos que compare lemmatização e steming, considerando a melhor forma de vetorização vista no item b)"""

# Dividindo os dados em conjuntos de treino e teste
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

# Caso 1: Lematização
# Aplicando lematização
X_train_lemmatized = X_train_raw.apply(preprocess_with_lemmatization)
X_test_lemmatized = X_test_raw.apply(preprocess_with_lemmatization)

# Aplicando Data Augmentation: Synonym Replacement e Random Insertion
X_augmented_lemmatized = pd.concat([
    X_train_lemmatized.apply(lambda x: synonym_replacement(x, n=2)),
    X_train_lemmatized.apply(lambda x: random_insertion(x, n=2))
])

# Combinando o dataset original com o aumentado
X_train_combined_lemmatized = pd.concat([X_train_lemmatized, X_augmented_lemmatized], axis=0)
y_train_combined_lemmatized = pd.concat([y_train] * 3, axis=0)

# Vetorização TF-IDF com lematização
vocab_tfidf_lemmatized, X_train_tfidf_lemmatized = tfidf_vectorizer(X_train_combined_lemmatized)

X_test_tfidf_lemmatized = np.zeros((len(X_test_lemmatized), len(vocab_tfidf_lemmatized)))
for i, document in enumerate(X_test_lemmatized):
    for word in document.split():
        if word in vocab_tfidf_lemmatized:
            X_test_tfidf_lemmatized[i, vocab_tfidf_lemmatized[word]] += 1

X_test_tfidf_lemmatized = X_test_tfidf_lemmatized * np.log((len(X_train_combined_lemmatized) + 1) / (np.sum(X_test_tfidf_lemmatized > 0, axis=0) + 1)) + 1

# Treinando o classificador com Lematização
clf_lemmatized = LogisticRegression()
clf_lemmatized.fit(X_train_tfidf_lemmatized, y_train_combined_lemmatized)

# Prevendo e calculando a acurácia com Lematização
y_pred_lemmatized = clf_lemmatized.predict(X_test_tfidf_lemmatized)
accuracy_lemmatized = accuracy_score(y_test, y_pred_lemmatized)

print("Classification Report com Lematização usando TF-IDF:")
print(classification_report(y_test, y_pred_lemmatized, zero_division=0))

# Caso 2: Stematização
# Aplicando stematização
X_train_stemmed = X_train_raw.apply(preprocess_with_stemming)
X_test_stemmed = X_test_raw.apply(preprocess_with_stemming)

# Aplicando Data Augmentation: Synonym Replacement e Random Insertion
X_augmented_stemmed = pd.concat([
    X_train_stemmed.apply(lambda x: synonym_replacement(x, n=2)),
    X_train_stemmed.apply(lambda x: random_insertion(x, n=2))
])

# Combinando o dataset original com o aumentado
X_train_combined_stemmed = pd.concat([X_train_stemmed, X_augmented_stemmed], axis=0)
y_train_combined_stemmed = pd.concat([y_train] * 3, axis=0)

# Vetorização TF-IDF com stematização
vocab_tfidf_stemmed, X_train_tfidf_stemmed = tfidf_vectorizer(X_train_combined_stemmed)

X_test_tfidf_stemmed = np.zeros((len(X_test_stemmed), len(vocab_tfidf_stemmed)))
for i, document in enumerate(X_test_stemmed):
    for word in document.split():
        if word in vocab_tfidf_stemmed:
            X_test_tfidf_stemmed[i, vocab_tfidf_stemmed[word]] += 1

X_test_tfidf_stemmed = X_test_tfidf_stemmed * np.log((len(X_train_combined_stemmed) + 1) / (np.sum(X_test_tfidf_stemmed > 0, axis=0) + 1)) + 1

# Treinando o classificador com Stematização
clf_stemmed = LogisticRegression()
clf_stemmed.fit(X_train_tfidf_stemmed, y_train_combined_stemmed)

# Prevendo e calculando a acurácia com Stematização
y_pred_stemmed = clf_stemmed.predict(X_test_tfidf_stemmed)
accuracy_stemmed = accuracy_score(y_test, y_pred_stemmed)

print("Classification Report com Stematização usando TF-IDF:")
print(classification_report(y_test, y_pred_stemmed, zero_division=0))



# Dados de acurácia
accuracy_data = {
    "Sem Pré-processamento": accuracy_raw,
    "Com Lemmatização": accuracy_lemmatized,
    "Com Stemming": accuracy_stemmed,
    "CountVectorizer": accuracy_count,
    "TF-IDF": accuracy_tfidf
}

# Criando gráfico de acurácia
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracy_data.keys()), y=list(accuracy_data.values()))
plt.title("Comparação de Acurácia")
plt.ylabel("Acurácia")
plt.xlabel("Método")
plt.ylim(0, 1)
plt.show()

# Dados para F1-score (lembrando que as chaves representam as classes 0, 1 e 2)
f1_data = {
    "Sem Pré-processamento": classification_report(y_test, y_pred_raw, output_dict=True),
    "Com Lemmatização": classification_report(y_test, y_pred_lemmatized, output_dict=True),
    "Com Stemming": classification_report(y_test, y_pred_stemmed, output_dict=True),
    "CountVectorizer": classification_report(y_test, y_pred_count, output_dict=True),
    "TF-IDF": classification_report(y_test, y_pred_tfidf, output_dict=True)
}

# Plotando F1-score por classe
plt.figure(figsize=(15, 8))
for method, scores in f1_data.items():
    classes = ['0', '1', '2']
    f1_scores = [scores[cls]['f1-score'] for cls in classes]
    plt.plot(classes, f1_scores, marker='o', label=method)

plt.title("Comparação de F1-score por Classe")
plt.ylabel("F1-score")
plt.xlabel("Classe")
plt.ylim(0, 1)
plt.legend()
plt.show()

# Plotando Precisão e Recall
precision_data = {}
recall_data = {}
for method, scores in f1_data.items():
    precision_data[method] = [scores[cls]['precision'] for cls in classes]
    recall_data[method] = [scores[cls]['recall'] for cls in classes]

plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
for method, precisions in precision_data.items():
    plt.plot(classes, precisions, marker='o', label=method)
plt.title("Comparação de Precisão por Classe")
plt.ylabel("Precisão")
plt.xlabel("Classe")
plt.ylim(0, 1)
plt.legend()

plt.subplot(1, 2, 2)
for method, recalls in recall_data.items():
    plt.plot(classes, recalls, marker='o', label=method)
plt.title("Comparação de Recall por Classe")
plt.ylabel("Recall")
plt.xlabel("Classe")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()