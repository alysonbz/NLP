from src.utils import load_movie_review_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def preprocess(text):
    text = text.lower()
    return text


corpus = load_movie_review_clean_dataset()

dataset = corpus.copy()
dataset['review'] = dataset['review'].apply(preprocess)

X = dataset['review']
y = dataset['sentiment']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)


vectorizer_count = CountVectorizer(stop_words="english")
vectorizer_tfidf = TfidfVectorizer(stop_words="english")

X_train_bow_count = vectorizer_count.fit_transform(X_train)
X_test_bow_count = vectorizer_count.transform(X_test)

X_train_bow_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_bow_tfidf = vectorizer_tfidf.transform(X_test)


clf_count = MultinomialNB()
clf_tfidf = MultinomialNB()

clf_count.fit(X_train_bow_count, y_train)
clf_tfidf.fit(X_train_bow_tfidf, y_train)


y_pred_count = clf_count.predict(X_test_bow_count)
y_pred_tfidf = clf_tfidf.predict(X_test_bow_tfidf)


print("===== COUNT VECTORIZER RESULTS =====")
print(classification_report(y_test, y_pred_count))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_count))

print("\n===== TF-IDF RESULTS =====")
print(classification_report(y_test, y_pred_tfidf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tfidf))