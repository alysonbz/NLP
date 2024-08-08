from src.utils import load_movie_review_dataset

# Import CountVectorizer
from sklearn.feature_extraction.text import ____

corpus = load_movie_review_dataset()
corpus = corpus['tagline'].dropna()

# Create CountVectorizer object
vectorizer = ____

# Generate matrix of word vectors
bow_matrix = _____

# Print the shape of bow_matrix
print(bow_matrix.shape)