import numpy as np 
import pandas as pd 
import re 
from collections import Counter

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.split()

def calc_tf(doc):
    counts = Counter(doc)
    total = len(doc)
    return {word: counts[word] / total for word in counts}

def calc_idf(corpus):
    n = len(corpus)
    df = {}

    for doc in corpus:
        for word in set(doc):
            df[word] = df.get(word, 0) + 1

    return {word: np.log(n / df[word]) for word in df}

def tf_idf_matrix(texts):
    corpus = [preprocess(t) for t in texts]

    idf = idf(corpus)

    tfidf_matrix = []
    vocabulary = sorted(list(idf.keys()))

    for doc in corpus:
        tf = calc_tf(doc)
        tfidf_vector = [tf.get(word, 0) * idf[word] for word in vocabulary]
        tfidf_matrix.append(tfidf_vector)

    return pd.DataFrame(tfidf_matrix, columns=vocabulary)

def load_synthetic_corpus():
    return [
        "The movie was great and very inspiring",
        "The movie was terrible and boring",
        "Great acting but the story was boring",
        "The film was inspiring and had great scenes",
        "Terrible movie with boring acting"
    ]


if __name__ == "__main__":
    corpus = load_synthetic_corpus()
    tfidf_df = tf_idf_matrix(corpus)
    print(tfidf_df)