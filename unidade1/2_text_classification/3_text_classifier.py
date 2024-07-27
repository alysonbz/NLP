from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from src.utils import load_movie_review_clean_dataset
from sklearn.naive_bayes import MultinomialNB

# Carregar o conjunto de dados
corpus = load_movie_review_clean_dataset()
X = corpus['review']
y = corpus['sentiment']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar um objeto CountVectorizer
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Ajustar e transformar X_train
X_train_bow = vectorizer.fit_transform(X_train)

# Transformar X_test
X_test_bow = vectorizer.transform(X_test)

# Imprimir o formato de X_train_bow e X_test_bow
print("Shape of X_train_bow:", X_train_bow.shape)
print("Shape of X_test_bow:", X_test_bow.shape)

# Criar um objeto MultinomialNB
clf = MultinomialNB()

# Treinar o classificador
clf.fit(X_train_bow, y_train)

# Medir a acurácia
accuracy = clf.score(X_test_bow, y_test)
print("A acurácia do classificador no conjunto de teste é %.3f" % accuracy)

# Prever o sentimento de uma crítica negativa
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("O sentimento previsto pelo classificador é %i" % prediction)
