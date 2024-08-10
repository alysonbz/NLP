from sklearn.feature_extraction.text import TfidfVectorizer

# Take the title text
title_text = ["The lion is the king of the jungle"]

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()


# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)