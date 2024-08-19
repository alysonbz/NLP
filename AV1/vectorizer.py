from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def manual_count_vectorizer(texts, vocab=None):
    if vocab is None:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)
        vocab = vectorizer.vocabulary_
    else:
        vectorizer = CountVectorizer(vocabulary=vocab)
        X = vectorizer.transform(texts)
    return X, vocab

def manual_tf_idf(texts, vocab=None):
    if vocab is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        vocab = vectorizer.vocabulary_
    else:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        X = vectorizer.transform(texts)
    return X, vocab
