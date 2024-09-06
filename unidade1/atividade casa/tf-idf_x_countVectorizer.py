#No arquivo tf-idf_x_countVectorizer.py faça um script que compare
# os acertos da classificação de sentimentos no dataset utilizando o mesmo classificador,
# mas considerando as duas formas de vetorização:
# TF-IDF e CountVectorizer. Utilize métricas como
# classification report e matriz de confusãopara uma análise adequada.
# Considere também aplicar pré-processamento.

from src.utils import load_movie_review_clean_dataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

corpus = load_movie_review_clean_dataset()


texts = corpus['review']
labels = corpus['sentiment']


def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

texts = texts.apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Pipeline com CountVectorizer
count_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),  # Inclui stopwords padrão do inglês
    ('classifier', MultinomialNB())
])

# Pipeline com TfidfVectorizer
tfidf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english')),  # Inclui stopwords padrão do inglês
    ('classifier', MultinomialNB())
])

count_pipeline.fit(X_train, y_train)
y_pred_count = count_pipeline.predict(X_test)

tfidf_pipeline.fit(X_train, y_train)
y_pred_tfidf = tfidf_pipeline.predict(X_test)

print("\nAvaliação com CountVectorizer:")
print(confusion_matrix(y_test, y_pred_count))
print(classification_report(y_test, y_pred_count))

print("\nAvaliação com TfidfVectorizer:")
print(confusion_matrix(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))
