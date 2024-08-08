# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from src.utils import load_movie_review_clean_dataset
from sklearn.naive_bayes import MultinomialNB

# Carregar o conjunto de dados
corpus = load_movie_review_clean_dataset()
X = corpus['review']
y = corpus['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar um objeto CountVectorizer
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Ajustar e transformar X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transformar X_test
X_test_bow = vectorizer.transform(X_test)

# Imprimir a forma de X_train_bow e X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)

# Criar um objeto MultinomialNB
clf = MultinomialNB()

# Ajustar o classificador
clf.fit(X_train_bow, y_train)

# Medir a precisão
accuracy = clf.score(X_test_bow, y_test)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Prever o sentimento de uma resenha negativa
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))
