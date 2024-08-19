from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
import nltk


nltk.download('punkt')
nltk.download('wordnet')

# Inicializar o lematizador e o stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# lematização e stemming
def lemmatize_text(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

def stem_text(text):
    tokens = word_tokenize(text)
    return ' '.join([stemmer.stem(token) for token in tokens])

# Funções de vetorização manual
def manual_count_vectorizer(text_data, vocab=None):
    vectorizer = CountVectorizer(vocabulary=vocab)
    X_counts = vectorizer.fit_transform(text_data)
    return X_counts.toarray(), vectorizer.vocabulary_

def manual_tf_idf(text_data, vocab=None):
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X_tfidf = vectorizer.fit_transform(text_data)
    return X_tfidf.toarray(), vectorizer.vocabulary_


redacao_treino = pd.read_csv(r'C:/Users/bianc/Downloads/Ciencia_de_dados/NLP/AV1/train_preprocessed.csv')


X = redacao_treino['aplicar']  # textos
y = redacao_treino['thematic_coherence']  # rótulos (alvo)

# lematização e stemming
X_lemma = X.apply(lemmatize_text)
X_stem = X.apply(stem_text)

# Dividir os dados em treino e teste
X_train_lemma, X_test_lemma, y_train, y_test = train_test_split(X_lemma, y, test_size=0.2, random_state=42)
X_train_stem, X_test_stem, y_train, y_test = train_test_split(X_stem, y, test_size=0.2, random_state=42)

# Lemmatization + CountVectorizer
X_train_count_lemma, vocab_lemma = manual_count_vectorizer(X_train_lemma)
X_test_count_lemma, _ = manual_count_vectorizer(X_test_lemma, vocab=vocab_lemma)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_count_lemma, y_train)
y_pred_lemma_count = rf_model.predict(X_test_count_lemma)

print("Lemmatization + CountVectorizer + Random Forest:")
print(classification_report(y_test, y_pred_lemma_count))

# Lemmatization + TF-IDF
X_train_tfidf_lemma, vocab_tfidf_lemma = manual_tf_idf(X_train_lemma)
X_test_tfidf_lemma, _ = manual_tf_idf(X_test_lemma, vocab=vocab_tfidf_lemma)

rf_model.fit(X_train_tfidf_lemma, y_train)
y_pred_lemma_tfidf = rf_model.predict(X_test_tfidf_lemma)

print("Lemmatization + TF-IDF + Random Forest:")
print(classification_report(y_test, y_pred_lemma_tfidf))

# Stemming + CountVectorizer
X_train_count_stem, vocab_stem = manual_count_vectorizer(X_train_stem)
X_test_count_stem, _ = manual_count_vectorizer(X_test_stem, vocab=vocab_stem)

rf_model.fit(X_train_count_stem, y_train)
y_pred_stem_count = rf_model.predict(X_test_count_stem)

print("Stemming + CountVectorizer + Random Forest:")
print(classification_report(y_test, y_pred_stem_count))

# Stemming + TF-IDF
X_train_tfidf_stem, vocab_tfidf_stem = manual_tf_idf(X_train_stem)
X_test_tfidf_stem, _ = manual_tf_idf(X_test_stem, vocab=vocab_tfidf_stem)

rf_model.fit(X_train_tfidf_stem, y_train)
y_pred_stem_tfidf = rf_model.predict(X_test_tfidf_stem)

print("Stemming + TF-IDF + Random Forest:")
print(classification_report(y_test, y_pred_stem_tfidf))

# palavras mais frequentes
def print_most_frequent_words(vocab, top_n=10):
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    for word, index in sorted_vocab[:top_n]:
        print(f'Word: {word}, Frequency: {index}')

print("Most frequent words with Lemmatization:")
print_most_frequent_words(vocab_lemma)

print("Most frequent words with Stemming:")
print_most_frequent_words(vocab_stem)
