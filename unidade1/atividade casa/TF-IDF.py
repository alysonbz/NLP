import math
import pandas as pd

def compute_tf_idf():
    corpus = [
        "eu amo praia",
        "o mingau gosta de dormir",
        "apollo gosta de carinho",
        "god is good all time",
        "eu tenho a melhor mãe do mundo"
    ]

    # tokenização
    corpus = [c.lower().split() for c in corpus]

    # vocabulário
    vocab = sorted(set(word for doc in corpus for word in doc))

    # TF
    tf = []
    for doc in corpus:
        doc_tf = {}
        for term in vocab:
            doc_tf[term] = doc.count(term) / len(doc)  # CORRETO
        tf.append(doc_tf)  # CORRETO

    # IDF
    N = len(corpus)
    idf = {}
    for term in vocab:
        df = sum(term in doc for doc in corpus)
        idf[term] = math.log((N + 1) / (df + 1)) + 1

    # TF-IDF
    tfidf = []
    for doc_tf in tf:
        tfidf_doc = {}
        for term in vocab:
            tfidf_doc[term] = doc_tf[term] * idf[term]
        tfidf.append(tfidf_doc)

    return pd.DataFrame(tfidf)


if __name__ == "__main__":
    matriz = compute_tf_idf()
    print("Matriz TF-IDF manual:")
    print(matriz)
