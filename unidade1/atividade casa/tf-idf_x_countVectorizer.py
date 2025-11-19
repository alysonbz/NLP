from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import re
from src.utils import load_movie_review_clean_dataset

# pré-processamento simples
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# carregando o dataset
df = load_movie_review_clean_dataset()

# aplicando pré-processamento
df["review"] = df["review"].apply(preprocess)

X = df["review"]
y = df["sentiment"]

# divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# vetorizadores
count_vect = CountVectorizer()
tfidf_vect = TfidfVectorizer()

X_train_count = count_vect.fit_transform(X_train)
X_test_count = count_vect.transform(X_test)

X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

# classificador
clf = LogisticRegression(max_iter=500)

# treino com CountVectorizer
clf.fit(X_train_count, y_train)
pred_count = clf.predict(X_test_count)

# treino com TF-IDF
clf.fit(X_train_tfidf, y_train)
pred_tfidf = clf.predict(X_test_tfidf)

print("\n**** RESULTADOS COUNT VECTORIZER ****")
print(classification_report(y_test, pred_count))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, pred_count))

print("\n**** RESULTADOS TF-IDF ****")
print(classification_report(y_test, pred_tfidf))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, pred_tfidf))

