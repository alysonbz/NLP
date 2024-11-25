# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from src.utils import load_movie_review_clean_dataset
from sklearn.naive_bayes import MultinomialNB

corpus = load_movie_review_clean_dataset()
X = corpus['review']
y = corpus['sentiment']
X_train,  X_test,y_train, y_test = _____

# Create a CountVectorizer object
vectorizer = ____(lowercase=____, stop_words=____)

# Fit and transform X_train
X_train_bow = vectorizer.____(____)

# Transform X_test
X_test_bow = vectorizer.____(____)

# Print shape of X_train_bow and X_test_bow
print(X_train_bow.shape)
print(X_test_bow.shape)


# Create a MultinomialNB object
clf = ____

# Fit the classifier
clf.____(____, ____)

# Measure the accuracy
accuracy = clf.score(____, ____)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

# Predict the sentiment of a negative review
review = "The movie was terrible. The music was underwhelming and the acting mediocre."
prediction = clf.predict(vectorizer.transform([review]))[0]
print("The sentiment predicted by the classifier is %i" % (prediction))