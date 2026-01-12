import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("resultados")
ANALYSIS_DIR = Path("analises")
PLOT_DIR = ANALYSIS_DIR / "plots"

ANALYSIS_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

def load_results():
    dfs = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".xlsx"):
            path = RESULTS_DIR / fname
            df = pd.read_excel(path)
            df["experiment"] = fname.replace(".xlsx", "")
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def aggregate(df):
    agg = df.groupby("experiment").agg({
        "BLEU": "mean",
        "BERTScore": "mean",
        "retrieval_time": "mean",
        "generation_time": "mean",
        "total_time": "mean"
    }).reset_index()

    meta_cols = ["retrieval", "embedding", "llm", "temp", "k", "chunk", "vectorstore"]
    meta = df.groupby("experiment")[meta_cols].first().reset_index()
    return agg.merge(meta, on="experiment")

def save_tables(agg):
    agg.to_excel(ANALYSIS_DIR / "agregados.xlsx", index=False)
    ranking = agg.sort_values("BERTScore", ascending=False)
    ranking.to_excel(ANALYSIS_DIR / "ranking.xlsx", index=False)
    print("Tabelas salvas!")

def plot_bar_with_labels(agg, group_col, metric, fname):
    plt.figure(figsize=(8,4))
    colors = sns.color_palette("tab10", n_colors=agg[group_col].nunique())
    means = agg.groupby(group_col)[metric].mean()
    ax = means.plot(kind="bar", color=colors)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.title(f"{metric} médio por {group_col}")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / fname)
    plt.close()

def plot_violin(df):
    plt.figure(figsize=(10,5))
    sns.violinplot(data=df, x="experiment", y="BERTScore", palette="tab10")
    plt.xticks(rotation=45, ha='right')
    plt.title("Distribuição do BERTScore por variação")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "violin_bertscore.png")
    plt.close()

def plot_heatmaps(df):
    for retrieval in df["retrieval"].unique():
        sub = df[df["retrieval"] == retrieval]
        pivot = sub.groupby(["k","chunk"])["BERTScore"].mean().unstack()
        plt.figure(figsize=(6,4))
        sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".3f")
        plt.title(f"Heatmap BERTScore (retrieval={retrieval})")
        plt.xlabel("chunk")
        plt.ylabel("k")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"heatmap_{retrieval}.png")
        plt.close()

def plot_tradeoff(agg):
    plt.figure(figsize=(7,5))
    for r in agg["retrieval"].unique():
        subset = agg[agg["retrieval"] == r]
        plt.scatter(subset["total_time"], subset["BERTScore"], s=80, label=r)

    plt.xlabel("Latência total média (s)")
    plt.ylabel("BERTScore médio")
    plt.title("Tradeoff latência x qualidade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "tradeoff_latency.png")
    plt.close()

def main():
    print("Carregando...")
    df = load_results()

    print("Agregando...")
    agg = aggregate(df)

    print("Salvando tabelas...")
    save_tables(agg)

    print("Gerando violin...")
    plot_violin(df)

    print("Gerando barras...")
    plot_bar_with_labels(agg, "retrieval", "BERTScore", "bert_por_retrieval.png")
    plot_bar_with_labels(agg, "llm", "BERTScore", "bert_por_llm.png")
    plot_bar_with_labels(agg, "embedding", "BERTScore", "bert_por_embedding.png")

    print("Gerando heatmaps...")
    plot_heatmaps(df)

    print("Gerando tradeoff...")
    plot_tradeoff(agg)

    print("OK! Tudo salvo em:", PLOT_DIR)

if __name__ == "__main__":
    main()
