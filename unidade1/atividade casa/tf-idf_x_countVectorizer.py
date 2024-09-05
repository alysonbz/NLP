import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.utils import load_movie_review_clean_dataset


def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    texto = ' '.join([lemmatizer.lemmatize(word) for word in texto.split() if word not in stop_words])
    return texto


corpus = load_movie_review_clean_dataset()

X = corpus['review']
y = corpus['sentiment']

X = X.apply(preprocessar_texto)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pip_count_vec = Pipeline([('vetorizador', CountVectorizer()), ('classificador', MultinomialNB())])

pip_tfidf = Pipeline([('vetorizador', TfidfVectorizer()), ('classificador', MultinomialNB())])

pip_count_vec.fit(x_train, y_train)
y_pred_count = pip_count_vec.predict(x_test)

pip_tfidf.fit(x_train, y_train)
y_pred_tfidf = pip_tfidf.predict(x_test)

print("CountVectorizer:")
print(classification_report(y_test, y_pred_count))  # 77% acuracia

print("TF-IDF:")
print(classification_report(y_test, y_pred_tfidf))  # 77% acurácia
