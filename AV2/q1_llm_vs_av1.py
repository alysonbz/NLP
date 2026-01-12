import os
import sys
import json
import time
import random
import pandas as pd
import numpy as np
import requests

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# CORREÇÃO DO IMPORT: adiciona a pasta "pai" (onde está AV1)
# ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from preprocessing import preprocess_dataset  # agora funciona


# =========================
# Baseline AV1 (Estatística + LogReg)
# =========================
def feats_estatistica(corpus: list[str]) -> np.ndarray:
    return np.array([[len(t.split()), sum(len(w) for w in t.split())] for t in corpus], dtype=np.float32)


def train_eval_baseline(df: pd.DataFrame, use_preprocess: bool = False,
                        use_stemming: bool = False, use_lemmatization: bool = False):
    if use_preprocess:
        df2 = preprocess_dataset(df, use_stemming=use_stemming, use_lemmatization=use_lemmatization)
        corpus = df2["premise_clean"].tolist()
    else:
        corpus = df["premise"].tolist()

    X = feats_estatistica(corpus)
    y = df["entailment_judgment"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "report": classification_report(y_test, pred, digits=4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }


# =========================
# LLM local via Ollama
# =========================
LABEL_MAP = {0: "none", 1: "entailment", 2: "paraphrase"}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

SYSTEM_PROMPT = """Você é um classificador de inferência textual (ASSIN) em português.
Recebe uma Premise e uma Hypothesis.
Responda APENAS com um dos rótulos: none, entailment ou paraphrase.

Definições:
- none: não há relação de implicação nem equivalência.
- entailment: a Premise implica a Hypothesis.
- paraphrase: as sentenças têm o mesmo significado essencial.
"""


def ollama_label(premise: str, hypothesis: str, model: str = "phi3.5") -> int:
    """
    Requer: Ollama instalado e modelo baixado.
    Teste no terminal: ollama run phi3.5
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Premise: {premise}\nHypothesis: {hypothesis}\nRótulo:"}
        ],
        "options": {"temperature": 0}
    }

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    text = r.json()["message"]["content"].strip().lower()
    text = text.replace(".", "").replace(":", "").strip()

    # normaliza saída
    if text not in INV_LABEL_MAP:
        for k in INV_LABEL_MAP:
            if k in text:
                return INV_LABEL_MAP[k]
        # fallback
        return 0

    return INV_LABEL_MAP[text]


def llm_eval_ollama(df: pd.DataFrame, sample_n: int = 150, seed: int = 42,
                    model: str = "phi3.5",
                    cache_path: str = "av2/outputs/q1_predictions.csv"):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # amostragem
    random.seed(seed)
    idxs = list(range(len(df)))
    random.shuffle(idxs)
    idxs = idxs[:sample_n]
    df_s = df.iloc[idxs].copy()

    # cache
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["sentence_pair_id", "y_true", "y_pred_llm"])

    cache_ids = set(cache["sentence_pair_id"].astype(int).tolist()) if len(cache) else set()

    preds = []
    for _, row in df_s.iterrows():
        spid = int(row["sentence_pair_id"])
        y_true = int(row["entailment_judgment"])

        if spid in cache_ids:
            y_pred = int(cache.loc[cache["sentence_pair_id"] == spid, "y_pred_llm"].iloc[0])
            preds.append((spid, y_true, y_pred))
            continue

        premise = str(row["premise"])
        hypothesis = str(row["hypothesis"])

        try:
            y_pred = ollama_label(premise, hypothesis, model=model)
        except Exception as e:
            print(f"[ERRO LLM] id={spid}: {e}")
            y_pred = 0

        preds.append((spid, y_true, y_pred))

        cache = pd.concat([cache, pd.DataFrame([{
            "sentence_pair_id": spid,
            "y_true": y_true,
            "y_pred_llm": y_pred
        }])], ignore_index=True)

        cache.to_csv(cache_path, index=False)
        time.sleep(0.05)

    y_true = np.array([t[1] for t in preds])
    y_pred = np.array([t[2] for t in preds])

    return {
        "n_samples": int(sample_n),
        "model": model,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, digits=4),
    }


if __name__ == "__main__":
    os.makedirs("av2/outputs", exist_ok=True)

    df = pd.read_parquet(os.path.join(BASE_DIR, "train-00000-of-00001.parquet"))

    # BASELINE: melhor caso da AV1 (Estatística sem preprocess)
    baseline = train_eval_baseline(df, use_preprocess=False)

    # LLM: avalia em amostra (150) para ficar rápido e gratuito
    llm = llm_eval_ollama(df, sample_n=150, model="phi3.5")

    out = {"baseline_av1_best": baseline, "llm_ollama": llm}

    with open("av2/outputs/q1_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== BASELINE (AV1 BEST) ===")
    print("Acc:", baseline["accuracy"], "Macro-F1:", baseline["macro_f1"])
    print(baseline["report"])

    print("\n=== LLM (Ollama) ===")
    print("Acc:", llm["accuracy"], "Macro-F1:", llm["macro_f1"])
    print(llm["report"])
