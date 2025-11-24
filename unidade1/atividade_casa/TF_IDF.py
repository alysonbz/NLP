import math
import pandas as pd

textos = [
    "this is not normal",
    "i am a dog",
    "i live in brazil",
    "brazil is a good country",
    "he lives in brazil",
]


def TF_IDF(corpus):
    N = len(corpus)

    # vocabulário 
    vocab = sorted(set(" ".join(corpus).split()))

    TF = []
    for doc in corpus:
        words = doc.split()
        doc_len = len(words)
        row = []
        for term in vocab:
            row.append(words.count(term) / doc_len)
        TF.append(row)

    # computar IDF
    def count_n_docs_with_term(term):
        return sum(term in doc.split() for doc in corpus)

    idfs = [math.log(N / (count_n_docs_with_term(term)+ 1))  for term in vocab]

    # matriz
    tfidf = []
    for row in TF:
        tfidf.append([tf * idf for tf, idf in zip(row, idfs)])

    return pd.DataFrame(tfidf, columns=vocab)



print(TF_IDF(textos))
