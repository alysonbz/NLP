import math


def load_corpus():
    return [
        "the lion is the king of the jungle",
        "lions have a lifespan of a decade",
        "the lion is an endangered species",
        "the jungle contains many species of animals",
        "a decade can change the balance of the jungle"
    ]

def compute_tf(doc):
    words = doc.split()
    tf = {}
    total_words = len(words)
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    for w in tf:
        tf[w] /= total_words
    return tf

def compute_idf(corpus):
    N = len(corpus)
    idf = {}
    all_words = set(word for doc in corpus for word in doc.split())

    for word in all_words:
        containing_docs = sum(word in doc.split() for doc in corpus)
        idf[word] = math.log((N + 1) / (containing_docs + 1)) + 1

    return idf

def compute_tfidf(corpus):
    tfs = [compute_tf(doc) for doc in corpus]
    idf = compute_idf(corpus)

    tfidf_matrix = []
    vocab = sorted(idf.keys())

    for tf in tfs:
        row = [tf.get(word, 0) * idf[word] for word in vocab]
        tfidf_matrix.append(row)

    return vocab, tfidf_matrix

if __name__ == "__main__":
    corpus = load_corpus()
    vocab, tfidf_matrix = compute_tfidf(corpus)

    print("VOCABULARY:")
    print(vocab)
    print("\nTF-IDF MATRIX:")
    for row in tfidf_matrix:
        print(row)
