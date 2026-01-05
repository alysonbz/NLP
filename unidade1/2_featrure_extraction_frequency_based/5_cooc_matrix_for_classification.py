from src.utils import load_movie_review_clean_dataset

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



corpus = load_movie_review_clean_dataset()


X = corpus['review']
y = corpus['sentiment']


X_train,  X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)



def _tokenize(text: str):
    return str(text).split()



def _pairs_in_window(tokens, window_size=2):
    # pares não ordenados: (a,b) == (b,a) -> simétrico e reduz dimensionalidade
    n = len(tokens)
    for i in range(n):
        j_max = min(n, i + window_size + 1)
        for j in range(i + 1, j_max):
            a, b = tokens[i], tokens[j]
            yield (a, b) if a <= b else (b, a)



def fit_pair_vocab(texts, window_size=2, max_features=20000, min_df=2):
    tf = Counter()  # frequência total
    df = Counter()  # em quantos docs

    for doc in texts:
        tokens = _tokenize(doc)
        pairs = list(_pairs_in_window(tokens, window_size))
        if not pairs:
            continue

        c = Counter(pairs)
        tf.update(c)
        for p in c.keys():
            df[p] += 1

    # seleciona os mais frequentes
    candidates = [p for p, d in df.items() if d >= min_df]
    candidates.sort(key=lambda p: (-tf[p], p))
    candidates = candidates[:max_features]

    return {p: i for i, p in enumerate(candidates)}


def transform_texts(texts, vocab, window_size=2):
    rows, cols, data = [], [], []

    for r, doc in enumerate(texts):
        tokens = _tokenize(doc)
        c = Counter(_pairs_in_window(tokens, window_size))

        for p, cnt in c.items():
            idx = vocab.get(p)
            if idx is not None:
                rows.append(r)
                cols.append(idx)
                data.append(cnt)

    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(texts), len(vocab)),
        dtype=np.float64
    )



# extração e classificação
pair_vocab = fit_pair_vocab(
    X_train,
    window_size=2,
    max_features=20000,
    min_df=2
)



X_train_cooc = transform_texts(X_train, pair_vocab, window_size=2)
X_test_cooc = transform_texts(X_test, pair_vocab, window_size=2)



tfidf = TfidfTransformer()
X_train_feat = tfidf.fit_transform(X_train_cooc)
X_test_feat = tfidf.transform(X_test_cooc)



clf = MultinomialNB()
clf.fit(X_train_feat, y_train)


y_pred = clf.predict(X_test_feat)


print("X_train:", X_train_feat.shape)
print("X_test :", X_test_feat.shape)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("\nMetricas:\n", classification_report(y_test, y_pred))
