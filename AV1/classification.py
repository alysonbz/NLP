from __future__ import annotations
from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from preprocessing import (
    preprocess_dataframe,
    preprocess_text
)
from atribute_extraction import FeatureExtractor
import matplotlib.pyplot as plt


def evaluate_feature_method(
    X_train: List[str],
    X_test: List[str],
    y_train,
    y_test,
    method: str,
    feature_extractor: FeatureExtractor,
):

    if method == "count":
        X_train_f = feature_extractor.fit_countvectorizer(X_train)
        X_test_f = feature_extractor.transform_countvectorizer(X_test)

    elif method == "tfidf":
        X_train_f = feature_extractor.fit_tfidf(X_train)
        X_test_f = feature_extractor.transform_tfidf(X_test)

    elif method == "word2vec":
        feature_extractor.train_word2vec(X_train)
        X_train_f = feature_extractor.get_word2vec_embeddings(X_train)
        X_test_f = feature_extractor.get_word2vec_embeddings(X_test)

    elif method == "cooc":
        cooc_matrix, vocab = feature_extractor.build_cooccurrence_matrix(X_train)
        X_train_f = cooc_matrix.values

        feature_extractor.train_word2vec(X_train)
        X_test_f = feature_extractor.get_word2vec_embeddings(X_test)

    else:
        raise ValueError("Método desconhecido.")

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_f, y_train)
    y_pred = clf.predict(X_test_f)

    acc = accuracy_score(y_test, y_pred)
    return acc


def run_all_experiments(df_path: str) -> pd.DataFrame:
    results = []

    df = pd.read_csv(df_path)
    X_raw = df["text"].astype(str)
    y = df["hatespeech_comb"].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_none = X_train_raw.tolist()
    X_test_none = X_test_raw.tolist()

    df_train = pd.DataFrame({"text": X_train_raw, "hatespeech_comb": y_train})
    df_test = pd.DataFrame({"text": X_test_raw, "hatespeech_comb": y_test})

    X_train_lemma, _ = preprocess_dataframe(df_train)
    X_test_lemma, _ = preprocess_dataframe(df_test)

    X_train_stem = [
        preprocess_text(t, use_stemming=True, use_lemmatization=False)
        for t in X_train_raw
    ]
    X_test_stem = [
        preprocess_text(t, use_stemming=True, use_lemmatization=False)
        for t in X_test_raw
    ]

    methods = ["count", "tfidf", "word2vec"]

    for m in methods:
        fx = FeatureExtractor(max_features=3000)

        acc_none = evaluate_feature_method(
            X_train_none, X_test_none, y_train, y_test, m, fx
        )

        fx = FeatureExtractor(max_features=3000)
        acc_lemma = evaluate_feature_method(
            X_train_lemma, X_test_lemma, y_train, y_test, m, fx
        )

        results.append({
            "Método": m,
            "Pré-processamento": "Nenhum",
            "Acurácia": acc_none
        })
        results.append({
            "Método": m,
            "Pré-processamento": "Lematização",
            "Acurácia": acc_lemma
        })

    best_method = "tfidf"  

    fx = FeatureExtractor(max_features=3000)
    acc_stem = evaluate_feature_method(
        X_train_stem, X_test_stem, y_train, y_test, best_method, fx
    )

    fx = FeatureExtractor(max_features=3000)
    acc_lemma = evaluate_feature_method(
        X_train_lemma, X_test_lemma, y_train, y_test, best_method, fx
    )

    results.append({
        "Método": best_method,
        "Pré-processamento": "Stemming",
        "Acurácia": acc_stem
    })
    results.append({
        "Método": best_method,
        "Pré-processamento": "Lematização",
        "Acurácia": acc_lemma
    })

    return pd.DataFrame(results)

def gerar_graficos(df_path: str, resultados: pd.DataFrame):

    print("\nGerando gráficos usando TF-IDF + Stemming (melhor modelo)...")

    df = pd.read_csv(df_path)
    X_raw = df["text"].astype(str)
    y = df["hatespeech_comb"].astype(int)

    X_stem = [
        preprocess_text(t, use_stemming=True, use_lemmatization=False)
        for t in X_raw
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X_stem, y, test_size=0.2, random_state=42, stratify=y
    )

    fx = FeatureExtractor(max_features=3000)
    X_train_vec = fx.fit_tfidf(X_train)
    X_test_vec = fx.transform_tfidf(X_test)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de Confusão – TF-IDF + Stemming")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.colorbar()

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i][j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig("matriz_confusao_tfidf_stemming.png", dpi=300)
    plt.close()

    print("✔ Matriz de confusão salva em: matriz_confusao_tfidf_stemming.png")

    plt.figure(figsize=(8, 5))
    plt.bar(
        range(len(resultados)),
        resultados["Acurácia"],
        color="skyblue"
    )
    plt.xticks(
        range(len(resultados)),
        resultados["Método"] + "\n" + resultados["Pré-processamento"],
        rotation=45,
        ha="right"
    )
    plt.ylabel("Acurácia")
    plt.title("Comparação das Acurácias por Método e Pré-processamento")
    plt.tight_layout()
    plt.savefig("acuracias_comparacao.png", dpi=300)
    plt.close()

    print("✔ Gráfico de barras salvo em: acuracias_comparacao.png")

if __name__ == "__main__":
    df_final = run_all_experiments("2019-05-28_portuguese_hate_speech_binary_classification.csv")
    print(df_final)
    df_final.to_csv("resultados_classificacao.csv", index=False)

    gerar_graficos("2019-05-28_portuguese_hate_speech_binary_classification.csv", df_final)
