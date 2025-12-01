import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_dataset
from atribute_extraction import AttributeExtractor


# ======================================================================
# Função: treina e avalia classificador
# ======================================================================

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


# ======================================================================
# Função: converte matriz de coocorrência em vetores por documento
# ======================================================================

def doc_vectors_from_cooccurrence(matrix, vocab_index, corpus):
    vectors = []

    for text in corpus:
        tokens = text.split()
        doc_vector = np.zeros(matrix.shape[0])

        for t in tokens:
            if t in vocab_index:
                doc_vector += matrix[vocab_index[t]]

        vectors.append(doc_vector)

    return np.array(vectors)


# ======================================================================
# Função: converte Word2Vec (por palavra) em vetores por documento
# ======================================================================

def doc_vectors_from_word2vec(model, corpus, vector_size=100):
    vectors = []

    for text in corpus:
        tokens = text.split()
        word_vecs = []

        for t in tokens:
            if t in model.wv:
                word_vecs.append(model.wv[t])

        if len(word_vecs) == 0:
            vectors.append(np.zeros(vector_size))
        else:
            vectors.append(np.mean(word_vecs, axis=0))

    return np.array(vectors)


# ======================================================================
# EXPERIMENTOS PRINCIPAIS (A, B, C)
# ======================================================================

def run_classification_experiments():
    df = pd.read_parquet("train-00000-of-00001.parquet")
    y = df["entailment_judgment"]

    print("\n==================== ITEM (A) ====================")

    # Raw corpus
    corpus_raw = df["premise"].tolist()

    # Preprocessed corpus
    df_clean = preprocess_dataset(df)
    corpus_clean = df_clean["premise_clean"].tolist()

    results_A = {}

    # ------------------------------------------------------
    # 1) CountVectorizer
    # ------------------------------------------------------
    print("\nTestando CountVectorizer...")

    ext_raw = AttributeExtractor(corpus_raw)
    X_raw, _ = ext_raw.count_vectorizer()
    acc_raw = train_and_evaluate(X_raw, y)

    ext_clean = AttributeExtractor(corpus_clean)
    X_clean, _ = ext_clean.count_vectorizer()
    acc_clean = train_and_evaluate(X_clean, y)

    results_A["CountVectorizer"] = (acc_raw, acc_clean)

    # ------------------------------------------------------
    # 2) TF-IDF
    # ------------------------------------------------------
    print("\nTestando TF-IDF...")

    X_raw, _ = ext_raw.tfidf_vectorizer()
    acc_raw = train_and_evaluate(X_raw, y)

    X_clean, _ = ext_clean.tfidf_vectorizer()
    acc_clean = train_and_evaluate(X_clean, y)

    results_A["TF-IDF"] = (acc_raw, acc_clean)

    # ------------------------------------------------------
    # 3) Coocorrência (corrigida)
    # ------------------------------------------------------
    print("\nTestando Coocorrência... (document-level)")

    matrix_raw, vocab_raw = ext_raw.cooccurrence_matrix()
    X_raw = doc_vectors_from_cooccurrence(matrix_raw, vocab_raw, corpus_raw)
    acc_raw = train_and_evaluate(X_raw, y)

    matrix_clean, vocab_clean = ext_clean.cooccurrence_matrix()
    X_clean = doc_vectors_from_cooccurrence(matrix_clean, vocab_clean, corpus_clean)
    acc_clean = train_and_evaluate(X_clean, y)

    results_A["Coocorrência"] = (acc_raw, acc_clean)

    # ------------------------------------------------------
    # 4) Word2Vec (corrigido)
    # ------------------------------------------------------
    print("\nTestando Word2Vec... (document-level)")

    emb_raw, _, w2v_raw = ext_raw.word2vec()
    X_raw = doc_vectors_from_word2vec(w2v_raw, corpus_raw)
    acc_raw = train_and_evaluate(X_raw, y)

    emb_clean, _, w2v_clean = ext_clean.word2vec()
    X_clean = doc_vectors_from_word2vec(w2v_clean, corpus_clean)
    acc_clean = train_and_evaluate(X_clean, y)

    results_A["Word2Vec"] = (acc_raw, acc_clean)

    # ------------------------------------------------------
    # 5) Estatística
    # ------------------------------------------------------
    print("\nTestando Estatísticas...")

    def get_feats(corpus):
        return np.array([[len(t.split()), sum(len(w) for w in t.split())] for t in corpus])

    X_raw = get_feats(corpus_raw)
    acc_raw = train_and_evaluate(X_raw, y)

    X_clean = get_feats(corpus_clean)
    acc_clean = train_and_evaluate(X_clean, y)

    results_A["Estatística"] = (acc_raw, acc_clean)

    # ------------------------------------------------------
    # Resultado item A
    # ------------------------------------------------------
    print("\n\nRESULTADOS DO ITEM (A):")
    print("---------------------------------------")
    for method, (raw_acc, clean_acc) in results_A.items():
        print(f"{method}: sem_preprocess={raw_acc:.4f} | com_preprocess={clean_acc:.4f}")

    # ==========================================================================
    # B) Selecionar melhor técnica
    # ==========================================================================
    print("\n==================== ITEM (B) ====================")

    best_method = max(results_A, key=lambda k: results_A[k][1])
    print(f"\n>>> Melhor técnica encontrada: **{best_method}**")

    # ==========================================================================
    # C) Lematização vs Stemming
    # ==========================================================================
    print("\n==================== ITEM (C) ====================")

    df_lemma = preprocess_dataset(df, use_lemmatization=True, use_stemming=False)
    df_stem = preprocess_dataset(df, use_lemmatization=False, use_stemming=True)

    corpus_lemma = df_lemma["premise_clean"].tolist()
    corpus_stem = df_stem["premise_clean"].tolist()

    ext_lemma = AttributeExtractor(corpus_lemma)
    ext_stem = AttributeExtractor(corpus_stem)

    if best_method == "CountVectorizer":
        X_lemma, _ = ext_lemma.count_vectorizer()
        X_stem, _ = ext_stem.count_vectorizer()

    elif best_method == "TF-IDF":
        X_lemma, _ = ext_lemma.tfidf_vectorizer()
        X_stem, _ = ext_stem.tfidf_vectorizer()

    elif best_method == "Coocorrência":
        mat_l, vocab_l = ext_lemma.cooccurrence_matrix()
        X_lemma = doc_vectors_from_cooccurrence(mat_l, vocab_l, corpus_lemma)

        mat_s, vocab_s = ext_stem.cooccurrence_matrix()
        X_stem = doc_vectors_from_cooccurrence(mat_s, vocab_s, corpus_stem)

    elif best_method == "Word2Vec":
        emb_l, _, w2v_l = ext_lemma.word2vec()
        X_lemma = doc_vectors_from_word2vec(w2v_l, corpus_lemma)

        emb_s, _, w2v_s = ext_stem.word2vec()
        X_stem = doc_vectors_from_word2vec(w2v_s, corpus_stem)

    else:
        X_lemma = get_feats(corpus_lemma)
        X_stem = get_feats(corpus_stem)

    acc_lemma = train_and_evaluate(X_lemma, y)
    acc_stem = train_and_evaluate(X_stem, y)

    print(f"\nACURÁCIA LEMMATIZAÇÃO: {acc_lemma:.4f}")
    print(f"ACURÁCIA STEMMING:     {acc_stem:.4f}")


# ======================================================================
# EXECUTAR
# ======================================================================

if __name__ == "__main__":
    run_classification_experiments()
