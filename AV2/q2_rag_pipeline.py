import os
import json
import glob
import re
import numpy as np
import requests

from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


# =========================
# Config
# =========================
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

OLLAMA_MODEL = "phi3.5"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_RAG = """Você é um assistente que responde perguntas usando APENAS o contexto fornecido.
Se o contexto não tiver a resposta, responda exatamente:
"Não encontrei essa informação nos documentos."
Responda de forma curta e objetiva.
"""


# =========================
# Estruturas
# =========================
@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


# =========================
# Leitura e chunking
# =========================
def load_docs(docs_dir: str):
    files = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    if not files:
        raise RuntimeError(f"Nenhum .txt encontrado em: {docs_dir}")
    docs = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(fp), f.read().strip()))
    return docs


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 80):
    """
    Chunking simples por caracteres (suficiente para docs curtos).
    """
    chunks = []
    i = 0
    while i < len(text):
        ch = text[i:i + chunk_size].strip()
        if ch:
            chunks.append(ch)
        i += max(1, chunk_size - overlap)
    return chunks


def build_chunks(docs, chunk_size=350, overlap=80):
    chunks = []
    for doc_id, text in docs:
        parts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for j, p in enumerate(parts):
            chunks.append(Chunk(doc_id=doc_id, chunk_id=j, text=p))
    return chunks


# =========================
# Retriever TF-IDF
# =========================
class TFIDFRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, query: str, top_k: int = 3):
        q = self.vectorizer.transform([query])
        sims = (self.X @ q.T).toarray().ravel()
        idxs = np.argsort(-sims)[:top_k]
        return [self.chunks[i] for i in idxs], sims[idxs].tolist()


def build_context(retrieved: list[Chunk]) -> str:
    ctx = []
    for c in retrieved:
        ctx.append(f"[{c.doc_id} | chunk {c.chunk_id}]\n{c.text}\n")
    return "\n".join(ctx)


# =========================
# LLM: Ollama
# =========================
def ollama_generate(question: str, context: str, model: str = OLLAMA_MODEL) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": f"Contexto:\n{context}\nPergunta: {question}\nResposta:"}
        ],
        "options": {"temperature": 0}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# =========================
# Avaliação
# =========================
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def contains_expected(answer: str, expected_keywords: list[str]) -> bool:
    ans = normalize(answer)
    return any(normalize(k) in ans for k in expected_keywords)


def run_eval(retriever: TFIDFRetriever, eval_set: list[dict], top_k: int = 3):
    rows = []
    y_true = []
    y_pred = []

    for item in eval_set:
        q = item["question"]
        expected = item["expected_keywords"]

        retrieved, scores = retriever.retrieve(q, top_k=top_k)
        context = build_context(retrieved)

        ans = ollama_generate(q, context)

        ok = contains_expected(ans, expected)

        rows.append({
            "question": q,
            "top_k": top_k,
            "retrieved_docs": [c.doc_id for c in retrieved],
            "retrieved_scores": scores,
            "answer": ans,
            "expected_keywords": expected,
            "ok": ok
        })

        y_true.append(1)
        y_pred.append(1 if ok else 0)

        print("\n==============================")
        print("Q:", q)
        print("Docs:", [c.doc_id for c in retrieved])
        print("A:", ans)
        print("OK:", ok)

    acc = accuracy_score(y_true, y_pred)
    return rows, float(acc)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # 1) Carregar docs e criar chunks
    docs = load_docs(DOCS_DIR)
    chunks = build_chunks(docs, chunk_size=350, overlap=80)

    # 2) Indexar e criar retriever
    retriever = TFIDFRetriever(chunks)

    # 3) Perguntas de avaliação (quantitativa + qualitativa)
    #    -> expected_keywords são termos que devem aparecer na resposta se ela for correta
    eval_set = [
        {
            "question": "Quais são as classes do entailment_judgment e o que cada uma significa?",
            "expected_keywords": ["none", "entailment", "paraphrase"]
        },
        {
            "question": "Qual coluna do ASSIN representa a similaridade semântica e qual é o intervalo dela?",
            "expected_keywords": ["relatedness_score", "1", "5"]
        },
        {
            "question": "Quais técnicas de extração de atributos foram testadas na AV1?",
            "expected_keywords": ["countvectorizer", "tf-idf", "coocorr", "word2vec", "estat"]
        },
        {
            "question": "Qual foi o melhor caso da AV1 em termos de acurácia e qual métrica ficou baixa por desbalanceamento?",
            "expected_keywords": ["0,7787", "macro-f1"]
        },
        {
            "question": "Explique rapidamente o objetivo do RAG.",
            "expected_keywords": ["recuper", "context", "alucin"]
        }
    ]

    # 4) Rodar avaliação
    rows, acc_proxy = run_eval(retriever, eval_set, top_k=3)

    # 5) Salvar resultados
    out = {
        "retriever": "TF-IDF",
        "llm": f"Ollama/{OLLAMA_MODEL}",
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "top_k": 3,
        "accuracy_proxy": acc_proxy,
        "notes": "Métrica proxy: considera correto quando a resposta contém palavras-chave esperadas.",
        "cases": rows
    }

    out_path = os.path.join(OUT_DIR, "q2_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== RESULTADO FINAL ===")
    print("Accuracy proxy:", acc_proxy)
    print("Arquivo salvo em:", out_path)
