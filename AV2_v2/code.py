#   pip install pandas scikit-learn matplotlib requests
#   ollama pull llama3.2


import os
import json
import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier


# Dataset
FAKE_PATH = r"C:\Data_Science\Aulas\6_Semestre\NLP\AV2\dataset\News_fake.csv"
REAL_PATH = r"C:\Data_Science\Aulas\6_Semestre\NLP\AV2\dataset\News_notFake.csv"


df_fake = pd.read_csv(FAKE_PATH)
df_real = pd.read_csv(REAL_PATH)


df_fake["label"] = 1  # Fake
df_real["label"] = 0  # Real


df = pd.concat([df_fake, df_real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


X = df["title"].astype(str)
y = df["label"].astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print("Dataset:", df.shape, "| Treino:", len(X_train), "| Teste:", len(X_test))
print("Proporção Fake (1):", float(y.mean()))



# Clássicos (seleção do melhor)
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)


models = {
    "LinearSVC": LinearSVC(),
    "LogReg": LogisticRegression(max_iter=2000),
    "MultinomialNB": MultinomialNB(),
    "SGD_hinge": SGDClassifier(loss="hinge", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


scores = {}
print("\nValidação cruzada (média 5-fold)")
for name, clf in models.items():
    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
    acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy").mean()
    scores[name] = {"f1": float(f1), "acc": float(acc)}
    print(f"{name:14s} | F1={f1:.4f} | Acc={acc:.4f}")

best_name = max(scores, key=lambda k: scores[k]["f1"])
print("\nMelhor (por F1):", best_name, scores[best_name])

best_pipe = Pipeline([("tfidf", vectorizer), ("clf", models[best_name])])
best_pipe.fit(X_train, y_train)

pred_classic = best_pipe.predict(X_test)

print("\nTESTE (Clássico)")
print("Accuracy:", accuracy_score(y_test, pred_classic))
print("F1:", f1_score(y_test, pred_classic))
print(classification_report(y_test, pred_classic, target_names=["Real", "Fake"], digits=4))

cm = confusion_matrix(y_test, pred_classic)
print("Matriz de confusão [ [Real->Real, Real->Fake], [Fake->Real, Fake->Fake] ]:\n", cm)

plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Clássico)")
plt.xticks([0, 1], ["Real", "Fake"])
plt.yticks([0, 1], ["Real", "Fake"])
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.tight_layout()
plt.show()

test_df = pd.DataFrame({"title": X_test.values, "y_true": y_test.values, "y_pred": pred_classic})
mis = test_df[test_df.y_true != test_df.y_pred].copy()

print("\nErros do clássico (qualitativo)")
if len(mis) == 0:
    print("Sem erros no teste.")
else:
    for _, row in mis.head(10).iterrows():
        print(f"- TRUE={row.y_true} PRED={row.y_pred} | {row.title}")



# LLM via Ollama (API open source)
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
CACHE_PATH = "llm_cache_q1_ollama.json"



def ollama_generate_json(system_prompt: str, user_prompt: str, timeout=120):
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "format": "json"  # força saída JSON (quando suportado)
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")



def normalize_llm_label(s: str):
    s = (s or "").strip().upper()
    if s in ["FAKE", "FALSA", "FALSO", "1", "LABEL_1"]:
        return 1
    if s in ["REAL", "VERDADEIRA", "VERDADEIRO", "0", "LABEL_0"]:
        return 0
    return None



def parse_json_like(text: str):
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None



def ollama_is_up():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False



if not ollama_is_up():
    print("\n[LLM] Ollama não está acessível em", OLLAMA_BASE)
    print("Suba o Ollama e rode novamente para gerar métricas da LLM.")
else:
    # Para controlar tempo, reduza N.
    N = len(X_test)

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            llm_cache = json.load(f)
    else:
        llm_cache = {}

    SYS = (
        "Você é um classificador de títulos de notícias em português do Brasil.\n"
        "Classifique o título como:\n"
        "- FAKE: boato/desinformação\n"
        "- REAL: factual\n\n"
        "Responda SOMENTE em JSON no formato:\n"
        "{\"label\":\"FAKE|REAL\",\"confidence\":0.0-1.0,\"reason\":\"curta\"}\n"
    )

    titles = X_test.values[:N]
    y_true = y_test.values[:N]

    llm_preds = []
    llm_reasons = []

    print("\nRodando LLM (Ollama) no teste (N=%d)" % N)

    for i, title in enumerate(titles):
        key = title
        if key in llm_cache:
            obj = llm_cache[key]
        else:
            raw = ollama_generate_json(SYS, f"Título: {title}")
            obj = parse_json_like(raw) or {"raw": raw}

            llm_cache[key] = obj
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(llm_cache, f, ensure_ascii=False, indent=2)

            time.sleep(0.05)

        label = normalize_llm_label(obj.get("label"))
        if label is None:
            raw = obj.get("raw", "") or json.dumps(obj, ensure_ascii=False)
            if "FAKE" in raw.upper():
                label = 1
            elif "REAL" in raw.upper():
                label = 0
            else:
                label = 0

        llm_preds.append(label)
        llm_reasons.append(obj.get("reason", ""))

        if (i + 1) % 20 == 0:
            print(f"[LLM] {i+1}/{N}")

    llm_preds = np.array(llm_preds, dtype=int)


    print("\n==== TESTE (LLM - Ollama) ====")
    print("Accuracy:", accuracy_score(y_true, llm_preds))
    print("F1 (Fake=1):", f1_score(y_true, llm_preds))
    print(classification_report(y_true, llm_preds, target_names=["Real", "Fake"], digits=4))


    cm_llm = confusion_matrix(y_true, llm_preds)
    print("Matriz de confusão (LLM):\n", cm_llm)

    # comparação qualitativa
    comp = pd.DataFrame({
        "title": titles,
        "y_true": y_true,
        "classic": pred_classic[:N],
        "llm": llm_preds,
        "llm_reason": llm_reasons
    })

    disag = comp[(comp["classic"] != comp["llm"]) | (comp["classic"] != comp["y_true"]) | (comp["llm"] != comp["y_true"])].copy()

    print("\nCasos para avaliação qualitativa (amostra)")
    for _, row in disag.head(15).iterrows():
        print("\nTítulo:", row["title"])
        print("Verdadeiro:", "Fake" if row["y_true"] == 1 else "Real")
        print("Clássico :", "Fake" if row["classic"] == 1 else "Real")
        print("LLM      :", "Fake" if row["llm"] == 1 else "Real")
        if row["llm_reason"]:
            print("Razão LLM:", row["llm_reason"])
