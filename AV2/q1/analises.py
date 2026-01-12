import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

def load_df(path):
    df = pd.read_csv(path)
    return df

def compute_metrics(df):
    """computar QWK + ACC"""
    df_valid = df.dropna(subset=["cohesion_pred"])
    y_true = df_valid["cohesion"].astype(int)
    y_pred = df_valid["cohesion_pred"].astype(int)

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    acc = accuracy_score(y_true, y_pred)

    return qwk, acc

def analyze_errors(df):
    df = df.copy()
    df["error"] = df["cohesion_pred"] - df["cohesion"]
    df["abs_error"] = df["error"].abs()

    worst = df.sort_values("abs_error", ascending=False).head(10)
    best = df.sort_values("abs_error").head(10)

    return best, worst

def plot_confusion(df):
    df_valid = df.dropna(subset=["cohesion_pred"])
    y_true = df_valid["cohesion"].astype(int)
    y_pred = df_valid["cohesion_pred"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5])

    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de Confusão (LLM)")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.xticks(range(5), [1,2,3,4,5])
    plt.yticks(range(5), [1,2,3,4,5])

    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i,j], ha="center", va="center")

    plt.savefig("cm.pdf", dpi=300)
    plt.show()

def plot_distribution(df):
    df_valid = df.dropna(subset=["cohesion_pred"])
    y_true = df_valid["cohesion"]
    y_pred = df_valid["cohesion_pred"]

    plt.figure(figsize=(6,4))
    plt.hist(y_true, bins=[1,2,3,4,5,6], alpha=0.6, label="Real")
    plt.hist(y_pred, bins=[1,2,3,4,5,6], alpha=0.6, label="Predito")
    plt.legend()
    plt.title("Distribuição — Real vs Predito")
    plt.savefig(
        "dist.pdf",
        dpi=300, )
    plt.show()

def show_examples(df):
    df = df.copy()
    df["error"] = df["cohesion_pred"] - df["cohesion"]
    df["abs_error"] = df["error"].abs()

    print("\n=== Melhores casos ===")
    print(df.sort_values("abs_error").head(5)[
        ["cohesion","cohesion_pred","cohesion_justification"]
    ])

    print("\n=== Piores casos ===")
    print(df.sort_values("abs_error", ascending=False).head(5)[
        ["cohesion","cohesion_pred","cohesion_justification"]
    ])

def main(path):
    df = load_df(path)

    print(">>> MÉTRICAS (LLM)")
    qwk, acc = compute_metrics(df)
    print(f"Acurácia = {acc:.4f}")
    print(f"QWK      = {qwk:.4f}")

    best, worst = analyze_errors(df)

    print("\n>>> MELHORES PREDIÇÕES")
    print(best[["cohesion","cohesion_pred","cohesion_justification"]])

    print("\n>>> PIORES PREDIÇÕES")
    print(worst[["cohesion","cohesion_pred","cohesion_justification"]])

    print("\n>>> MATRIZ DE CONFUSÃO")
    plot_confusion(df)

    print("\n>>> DISTRIBUIÇÃO REAL vs PREDITO")
    plot_distribution(df)

    print("\n>>> EXEMPLOS QUALITATIVOS")
    show_examples(df)

if __name__ == "__main__":
    main("results.csv")
