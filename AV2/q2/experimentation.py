from utils.data_loader import load_txt_folder
from utils.vector_index import build_vector_store
from utils.metrics import compute_bleu, compute_bertscore
from utils.rag_pipeline import rag_query, build_sparse_retriever


import pandas as pd
from pathlib import Path
import nltk
import re

nltk.download("punkt")

# ===== CONFIG =====
DATA_PATH = Path("dados")
Q_PATH = Path("perguntas_rag.xlsx")
RESULTS_PATH = Path("resultados")
RESULTS_PATH.mkdir(exist_ok=True, parents=True)


# =======================================================
# VARIAÇÕES EXPERIMENTAIS
# =======================================================

EXPERIMENTS = {
    "dense_1": {
        "embedding": "google",
        "llm": "gpt-4o-mini",
        "temp": 0.0,
        "k": 2,
        "chunk": 300,
        "overlap": 100,
        "vectorstore": "chroma",
        "retrieval": "dense",
    },
    "dense_2": {
        "embedding": "openai",
        "llm": "gemini-2.5-flash",
        "temp": 0.7,
        "k": 4,
        "chunk": 600,
        "overlap": 100,
        "vectorstore": "faiss",
        "retrieval": "dense",
    },
    "dense_3": {
        "embedding": "huggingface",
        "hf_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm": "gpt-4o-mini",
        "temp": 0.3,
        "k": 4,
        "chunk": 400,
        "overlap": 100,
        "vectorstore": "chroma",
        "retrieval": "dense",
    },
    "dense_4": {
        "embedding": "openai",
        "llm": "gemini-2.5-flash",
        "temp": 1.0,
        "k": 6,
        "chunk": 800,
        "overlap": 100,
        "vectorstore": "faiss",
        "retrieval": "dense",
    },
    "sparse_1": {
        "embedding": "google",
        "llm": "gpt-4o-mini",
        "temp": 0.0,
        "k": 4,
        "chunk": 300,
        "overlap": 100,
        "vectorstore": None,
        "retrieval": "sparse",
    },
    "sparse_2": {
        "embedding": "openai",
        "llm": "gemini-2.5-flash",
        "temp": 0.7,
        "k": 6,
        "chunk": 300,
        "overlap": 100,
        "vectorstore": None,
        "retrieval": "sparse",
    },
    "sparse_3": {
        "embedding": "huggingface",
        "hf_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "llm": "gpt-4o-mini",
        "temp": 0.5,
        "k": 4,
        "chunk": 500,
        "overlap": 100,
        "vectorstore": None,
        "retrieval": "sparse",
    },
    "sparse_4": {
        "embedding": "openai",
        "llm": "gemini-2.5-flash",
        "temp": 0.0,
        "k": 2,
        "chunk": 700,
        "overlap": 100,
        "vectorstore": None,
        "retrieval": "sparse",
    },
    "hybrid_1": {
        "embedding": "google",
        "llm": "gpt-4o-mini",
        "temp": 0.0,
        "k": 4,
        "chunk": 300,
        "overlap": 100,
        "vectorstore": "chroma",
        "retrieval": "hybrid",
    },
    "hybrid_2": {
        "embedding": "openai",
        "llm": "gemini-2.5-flash",
        "temp": 0.0,
        "k": 6,
        "chunk": 600,
        "overlap": 100,
        "vectorstore": "faiss",
        "retrieval": "hybrid",
    },
    "hybrid_3": {
        "embedding": "huggingface",
        "hf_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm": "gpt-4o-mini",
        "temp": 0.4,
        "k": 4,
        "chunk": 400,
        "overlap": 100,
        "vectorstore": "chroma",
        "retrieval": "hybrid",
    },
    "hybrid_4": {
        "embedding": "openai",
        "llm": "gemini-2.5-flash",
        "temp": 0.9,
        "k": 8,
        "chunk": 800,
        "overlap": 100,
        "vectorstore": "faiss",
        "retrieval": "hybrid",
    },
}

def sanitize(s):
    return re.sub(r"[^a-zA-Z0-9\-]+", "-", s)

def main():

    print("== Carregando documentos ==")
    docs = load_txt_folder(str(DATA_PATH))

    print("== Construindo sparse retriever ==")
    sparse = build_sparse_retriever(docs)

    print("== Carregando perguntas ==")
    df_questions = pd.read_excel(Q_PATH)
    perguntas = df_questions["perguntas"].astype(str).tolist()
    esperadas = df_questions["resposta_esperada"].astype(str).tolist()

    for name, cfg in EXPERIMENTS.items():

        emb = cfg["embedding"]
        llm = cfg["llm"]
        temp = cfg["temp"]
        k = cfg["k"]
        csize = cfg["chunk"]
        overlap = cfg["overlap"]
        vs_type = cfg["vectorstore"]
        retrieval = cfg["retrieval"]

        if overlap >= csize:
            print(f"[SKIP] Overlap >= chunk em {name}")
            continue

        filename = f"{sanitize(name)}.xlsx"
        path_out = RESULTS_PATH / filename

        if path_out.exists():
            print(f"[SKIP] já existe: {path_out}")
            continue

        print("\n== Novo experimento ==")
        print(f"Var:       {name}")
        print(f"Retrieval: {retrieval}")
        print(f"Embedding: {emb}")
        print(f"LLM:       {llm}")
        print(f"Temp:      {temp}")
        print(f"k:         {k}")
        print(f"chunk:     {csize}")
        print(f"bd:        {vs_type}")
        print("-------------------------")

        # ========== dense/hybrid precisam de vectorstore ==========
        if retrieval in ("dense", "hybrid"):
            vectorstore = build_vector_store(
                docs,
                embedding_type=emb,
                vectorstore_type=vs_type,
                chunk_size=csize,
                chunk_overlap=overlap,
                persist_directory=None,
                collection_name=f"{name}_{sanitize(emb)}_{csize}"
            )
        else:
            vectorstore = None

        registros = []

        for pergunta, esperado in zip(perguntas, esperadas):

            resp = rag_query(
                vectorstore,
                pergunta,
                llm_provider="google" if "gemini" in llm else "openai",
                llm_model_name=llm,
                temperature=temp,
                k=k,
                retrieval_mode=retrieval,
                sparse_retriever=sparse
            )

            res = resp["answer"]
            retrieval_time = resp["retrieval_time"]
            generation_time = resp["generation_time"]
            total_time = resp["total_time"]

            bleu = compute_bleu(esperado, res)
            bert = compute_bertscore(esperado, res)

            registros.append({
                "pergunta": pergunta,
                "esperada": esperado,
                "gerada": res,
                "BLEU": bleu,
                "BERTScore": bert,
                "var": name,
                "retrieval": retrieval,
                "embedding": emb,
                "llm": llm,
                "temp": temp,
                "k": k,
                "chunk": csize,
                "overlap": overlap,
                "vectorstore": vs_type,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
            })

            print(f"Pergunta: {pergunta}")
            print(f"BLEU: {bleu:.4f} | BERT: {bert:.4f}")
            print("------------------------")

        pd.DataFrame(registros).to_excel(path_out, index=False)
        print(f"[OK] Salvo: {path_out}")


if __name__ == "__main__":
    main()
