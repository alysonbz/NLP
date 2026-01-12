import os
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
import numpy as np
# ======================================================
# Configurações
# ======================================================
DOCS_DIR = "docs"
TOP_K = 3

QUESTIONS = [
    {
        "q": "Quais são as classes do entailment_judgment e o que cada uma significa?",
        "keywords": ["none", "entailment", "paraphrase"]
    },
    {
        "q": "Qual coluna do ASSIN representa a similaridade semântica e qual é o intervalo dela?",
        "keywords": ["relatedness_score", "1", "5"]
    },
    {
        "q": "Quais técnicas de extração de atributos foram testadas na AV1?",
        "keywords": ["countvectorizer", "tf-idf", "coocorrência", "word2vec", "estatística"]
    }
]

# ======================================================
# Utilidades
# ======================================================
def load_docs():
    docs = {}
    for f in os.listdir(DOCS_DIR):
        with open(os.path.join(DOCS_DIR, f), "r", encoding="utf-8") as file:
            docs[f] = file.read()
    return docs


def chunk_fixed(text, size=300):
    return [text[i:i+size] for i in range(0, len(text), size)]


def chunk_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 20]


def retrieve(chunks, query, k=3):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(chunks + [query])
    sims = cosine_similarity(X[-1], X[:-1])[0]
    top_idx = sims.argsort()[-k:][::-1]
    return [chunks[i] for i in top_idx]


def hybrid_retrieve(chunks, query, k=3):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(chunks + [query])
    sims = cosine_similarity(X[-1], X[:-1])[0]

    keywords = query.lower().split()
    bonus = []

    for c in chunks:
        score = sum(1 for w in keywords if w in c.lower())
        bonus.append(score)

    # 🔧 CORREÇÃO AQUI
    bonus = np.array(bonus)

    if bonus.max() > 0:
        bonus_norm = bonus / (bonus.max() + 1)
    else:
        bonus_norm = bonus

    final_score = sims + 0.1 * bonus_norm

    top_idx = final_score.argsort()[-k:][::-1]
    return [chunks[i] for i in top_idx]


def ask_llm(context, question):
    prompt = f"""Use apenas o contexto abaixo para responder.

Contexto:
{context}

Pergunta:
{question}
"""
    response = ollama.chat(
        model="phi3.5",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def evaluate(answer, keywords):
    answer = answer.lower()
    return all(k.lower() in answer for k in keywords)

# ======================================================
# PIPELINE 2 — Sentence-based chunking
# ======================================================
def pipeline_sentence_chunking(docs):
    chunks = []
    for t in docs.values():
        chunks.extend(chunk_sentences(t))

    results = []
    for q in QUESTIONS:
        retrieved = retrieve(chunks, q["q"], TOP_K)
        context = "\n".join(retrieved)
        answer = ask_llm(context, q["q"])
        ok = evaluate(answer, q["keywords"])
        results.append(ok)

    return sum(results) / len(results)

# ======================================================
# PIPELINE 3 — Hybrid retrieval
# ======================================================
def pipeline_hybrid(docs):
    chunks = []
    for t in docs.values():
        chunks.extend(chunk_fixed(t))

    results = []
    for q in QUESTIONS:
        retrieved = hybrid_retrieve(chunks, q["q"], TOP_K)
        context = "\n".join(retrieved)
        answer = ask_llm(context, q["q"])
        ok = evaluate(answer, q["keywords"])
        results.append(ok)

    return sum(results) / len(results)

# ======================================================
# EXECUÇÃO
# ======================================================
if __name__ == "__main__":
    docs = load_docs()

    acc_sentence = pipeline_sentence_chunking(docs)
    acc_hybrid = pipeline_hybrid(docs)

    output = {
        "pipeline_2_sentence_chunking": acc_sentence,
        "pipeline_3_hybrid_retrieval": acc_hybrid
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/q2_variants_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n=== RESULTADOS DOS PIPELINES ADICIONAIS ===")
    print(output)
