import numpy as np
import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Záéíóúàãõç\s]', '', text)
    return text.strip()

def load_corpus():
    """
    Retorna um corpus sintético de 5 linhas.
    """
    corpus = [
        "Eu adoro aprender NLP",
        "Machine Learning é incrível",
        "Eu gosto de aprender novas coisas",
        "Aprender é uma jornada contínua",
        "Machine Learning ajuda a resolver problemas"
    ]
    return [preprocess(t) for t in corpus]


def calc_tf(document):
    words = document.split()
    tf = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    total = len(words)
    return {w: count / total for w, count in tf.items()}


def calc_idf(corpus):
    N = len(corpus)
    word_doc_count = {}

    for doc in corpus:
        words = set(doc.split())
        for w in words:
            word_doc_count[w] = word_doc_count.get(w, 0) + 1

    idf = {w: np.log(N / word_doc_count[w]) for w in word_doc_count}
    return idf


def calc_tfidf(corpus):
    tfs = [calc_tf(doc) for doc in corpus]
    idf = calc_idf(corpus)

    vocabulary = sorted(idf.keys())
    tfidf_matrix = np.zeros((len(corpus), len(vocabulary)))

    for i, tf_doc in enumerate(tfs):
        for j, word in enumerate(vocabulary):
            tfidf_matrix[i][j] = tf_doc.get(word, 0) * idf[word]

    return vocabulary, tfidf_matrix


if __name__ == "__main__":
    corpus = load_corpus()
    vocab, tfidf = calc_tfidf(corpus)

    print("Vocabulário:")
    print(vocab)
    print("\nMatriz TF-IDF (manual):")
    print(tfidf)
