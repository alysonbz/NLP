import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import preprocess_text
from vectorizer import count_vectorizer, tfidf_vectorizer

# Função para aplicar o pré-processamento em um conjunto de textos
def preprocess_texts(texts, use_stemming=True, use_lemmatization=False):
    return [preprocess_text(text, use_stemming, use_lemmatization) for text in texts]

# Função para treinar e avaliar o classificador
def evaluate_classifier(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Carregar o dataset e processar
data = pd.read_csv('C:\\Users\\laura\\Downloads\\buscape_processado.csv')
data = data.dropna(subset=['review_text', 'polarity'])
texts = data['review_text']
labels = data['polarity']

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Comparar TF-IDF com e sem pré-processamento
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
accuracy_no_preprocessing = evaluate_classifier(X_train_tfidf, X_test_tfidf, y_train, y_test)

preprocessed_texts = preprocess_texts(texts.tolist(), use_lemmatization=True)
X_train_preprocessed, X_test_preprocessed = train_test_split(preprocessed_texts, test_size=0.2, random_state=42)
X_train_tfidf_preprocessed = tfidf_vectorizer.fit_transform(X_train_preprocessed)
X_test_tfidf_preprocessed = tfidf_vectorizer.transform(X_test_preprocessed)
accuracy_lemmatize = evaluate_classifier(X_train_tfidf_preprocessed, X_test_tfidf_preprocessed, y_train, y_test)

# Comparar CountVectorizer e TF-IDF com pré-processamento
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train_preprocessed)
X_test_count = count_vectorizer.transform(X_test_preprocessed)
accuracy_count_vectorizer = evaluate_classifier(X_train_count, X_test_count, y_train, y_test)

# Comparar lemmatização e stemming com a melhor vetorização (TF-IDF)
stemmed_texts = preprocess_texts(texts.tolist(), use_stemming=True)
X_train_stemmed, X_test_stemmed = train_test_split(stemmed_texts, test_size=0.2, random_state=42)
X_train_tfidf_stemmed = tfidf_vectorizer.fit_transform(X_train_stemmed)
X_test_tfidf_stemmed = tfidf_vectorizer.transform(X_test_stemmed)
accuracy_stemming = evaluate_classifier(X_train_tfidf_stemmed, X_test_tfidf_stemmed, y_train, y_test)

# Resultados
print(f"TF-IDF sem pré-processamento: {accuracy_no_preprocessing:.4f}")
print(f"TF-IDF com lemmatização: {accuracy_lemmatize:.4f}")
print(f"CountVectorizer com lemmatização: {accuracy_count_vectorizer:.4f}")
print(f"TF-IDF com stemming: {accuracy_stemming:.4f}")