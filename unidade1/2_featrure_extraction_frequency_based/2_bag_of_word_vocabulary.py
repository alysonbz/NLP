import pandas as pd
from src.utils import load_movie_review_dataset

# Import CountVectorizer
from sklearn.feature_extraction.text import ____

corpus = load_movie_review_dataset()
corpus = corpus['tagline'].dropna()

# Create CountVectorizer object
vectorizer = ____

# Generate matrix of word vectors
bow_matrix = _____

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Map the column names to vocabulary
bow_df.columns = _____

# Print bow_df
print(bow_df)