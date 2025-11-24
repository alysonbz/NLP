from src.utils import load_movie_review_clean_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

corpus = load_movie_review_clean_dataset()

X = corpus['review'].values
y = corpus['sentiment'].values

labels = np.unique(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

print("TF-IDF Vectorizer - Training data shape:", X_train_tfidf.shape)
print("TF-IDF Vectorizer - Test data shape:", X_test_tfidf.shape)

print("Count Vectorizer - Training data shape:", X_train_count.shape)
print("Count Vectorizer - Test data shape:", X_test_count.shape)

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
accuracy_tfidf = nb_tfidf.score(X_test_tfidf, y_test)
print("Naive Bayes with TF-IDF accuracy:", accuracy_tfidf)
print("Classification Report for TF-IDF Vectorizer:")
print(classification_report(y_test, nb_tfidf.predict(X_test_tfidf)))
# Confusion Matrix for TF-IDF
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)
cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)
plt.figure(figsize=(6,4))
sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - TF-IDF Vectorizer')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_tfidf.png')
plt.show()

nb_count = MultinomialNB()
nb_count.fit(X_train_count, y_train)
accuracy_count = nb_count.score(X_test_count, y_test)
print("Naive Bayes with Count Vectorizer accuracy:", accuracy_count)
print("Classification Report for Count Vectorizer:")
print(classification_report(y_test, nb_count.predict(X_test_count)))
# Confusion Matrix for Count Vectorizer
y_pred_count = nb_count.predict(X_test_count)
cm_count = confusion_matrix(y_test, y_pred_count)
plt.figure(figsize=(6,4))
sns.heatmap(cm_count, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Count Vectorizer')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_count.png')
plt.show()
