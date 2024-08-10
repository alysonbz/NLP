from src.utils import load_movie_review_clean_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import string
import re
corpus = load_movie_review_clean_dataset()

def preprocess_text(text):
    text.lower()
    text = text.translate(str.maketrans('','', string.ponctuation))
    text = re.sub(r'\d+','', text)
    text = text.strip()
    return text

corpus['processed_overview'] = corpus['overview'].apply(preprocess_text)
x = corpus['processed_overview']
corpus['sentiment'] = np.random.choice(['positive','negative'], len(corpus))
y = corpus['sentiment']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# CountVectorizer
count_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])

count_pipeline.fit(X_train, y_train)
y_pred_count = count_pipeline.predict(X_test)

# TF-IDF Vectorizer
tfidf_pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

tfidf_pipeline.fit(X_train, y_train)
y_pred_tfidf = tfidf_pipeline.predict(X_test)

# 6. Avaliação
print("CountVectorizer Classification Report")
print(classification_report(y_test, y_pred_count, target_names=le.classes_))

print("TF-IDF Vectorizer Classification Report")
print(classification_report(y_test, y_pred_tfidf, target_names=le.classes_))

print("CountVectorizer Confusion Matrix")
print(confusion_matrix(y_test, y_pred_count))

print("TF-IDF Vectorizer Confusion Matrix")
print(confusion_matrix(y_test, y_pred_tfidf))