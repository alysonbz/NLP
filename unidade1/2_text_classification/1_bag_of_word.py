from src.utils import load_movie_review_dataset

corpus = load_movie_review_dataset()
corpus = corpus['tagline'].dropna()

# Create CountVectorizer object

# Generate matrix of word vectors

# Print the shape of bow_matrix
print(bow_matrix.shape)