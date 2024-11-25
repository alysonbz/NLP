from src.utils import load_movie_review_clean_dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt


corpus= load_movie_review_clean_dataset()

X = corpus['review']
y = corpus['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def train_and_evaluate(vectorizer):
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', MultinomialNB())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    conf_matrix = confusion_matrix(y_test, y_pred)

    return acc, report, conf_matrix

# Avaliar com CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
count_acc, count_report, count_conf_matrix = train_and_evaluate(count_vectorizer)

# Avaliar com TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_acc, tfidf_report, tfidf_conf_matrix = train_and_evaluate(tfidf_vectorizer)

print("\n--- CountVectorizer ---")
print(f"Accuracy: {count_acc}")
print(count_report)
sns.heatmap(count_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("CountVectorizer Confusion Matrix")
plt.show()

print("\n--- TF-IDF ---")
print(f"Accuracy: {tfidf_acc}")
print(tfidf_report)
sns.heatmap(tfidf_conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("TF-IDF Confusion Matrix")
plt.show()