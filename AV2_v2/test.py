import os
import json
import glob
import numpy as np
import requests
import textwrap

# CONFIG
PROJECT_ROOT = r"C:\Data_Science\Aulas\6_Semestre\NLP\AV2\AV2_v2"
CACHE_DIR = os.path.join(PROJECT_ROOT, ".rag_cache")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")

TOP_K = 4
SHOW_TEXT_CHARS = 450

# HELPERS
def find_latest_cache_files(cache_dir: str):
    emb_files = sorted(glob.glob(os.path.join(cache_dir, "emb_*.npy")), key=os.path.getmtime, reverse=True)
    meta_files = sorted(glob.glob(os.path.join(cache_dir, "meta_*.json")), key=os.path.getmtime, reverse=True)
    if not emb_files or not meta_files:
        raise FileNotFoundError(
            f"Não encontrei cache em {cache_dir}. Rode o chat_interativo.py ao menos uma vez para gerar o índice."
        )
    # assume que o mais recente é o par correto
    return emb_files[0], meta_files[0]

def l2_normalize(mat: np.ndarray, eps: float = 1e-12):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + eps)

def ollama_embed(text: str, base_url: str, model: str) -> np.ndarray:
    url = base_url.rstrip("/") + "/api/embed"
    payload = {"model": model, "input": text}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Ollama retorna lista de embeddings em "embeddings"
    emb = np.array(data["embeddings"][0], dtype=np.float32)
    return emb



def cosine_topk(E_norm: np.ndarray, q: np.ndarray, k: int):
    q = q.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    scores = E_norm @ q
    idx = np.argsort(scores)[::-1][:k]
    return idx, scores[idx]



def safe_get(meta_item: dict, keys):
    for k in keys:
        if k in meta_item:
            return meta_item[k]
    return None


# MAIN
def main():
    emb_path, meta_path = find_latest_cache_files(CACHE_DIR)
    print(f"Embeddings: {emb_path}")
    print(f"Metadados : {meta_path}\n")

    E = np.load(emb_path)  # pode estar normalizado ou não (vamos normalizar de qualquer forma)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Se o meta for dict com campos, tenta achar a lista
    # (no seu projeto pode ser lista direta; mantemos robusto)
    if isinstance(meta, dict) and "meta" in meta:
        meta_list = meta["meta"]
        meta_info = {k: v for k, v in meta.items() if k != "meta"}
    else:
        meta_list = meta
        meta_info = {}

    print("=== INFO DO ÍNDICE ===")
    print(f"Qtd vetores (chunks): {E.shape[0]}")
    print(f"Dimensão do embedding: {E.shape[1]}")
    if meta_info:
        print("Info adicional (cache):")
        for k, v in meta_info.items():
            print(f"  - {k}: {v}")
    print()

    # Normaliza para cosseno
    E_norm = l2_normalize(E)

    # Mostra alguns exemplos
    print("=== EXEMPLOS (3 CHUNKS) ===")
    for i in range(min(3, len(meta_list))):
        item = meta_list[i]
        text = safe_get(item, ["text", "chunk", "content"])  # depende do seu schema
        src  = safe_get(item, ["source", "file", "doc"])
        print(f"\nChunk #{i}")
        if src:
            print(f"Fonte: {src}")
        if text:
            print(textwrap.shorten(text.replace("\n", " "), width=SHOW_TEXT_CHARS, placeholder=" ..."))
        print(f"Vector[0:8]: {E[i, :8]}")

    # Query interativa
    print("\n=== BUSCA (TOP-K) ===")
    query = input("Digite uma pergunta para testar a recuperação: ").strip()
    if not query:
        query = "Explique o que é RAG e qual o papel do banco vetorial."
        print(f"Usando query default: {query}")

    q_emb = ollama_embed(query, OLLAMA_BASE_URL, EMBED_MODEL)
    idx, sc = cosine_topk(E_norm, q_emb, TOP_K)

    print(f"\nModelo de embedding (Ollama): {EMBED_MODEL}")
    print(f"Top-{TOP_K} resultados:\n")
    for rank, (i, s) in enumerate(zip(idx, sc), start=1):
        item = meta_list[int(i)]
        text = safe_get(item, ["text", "chunk", "content"]) or ""
        src  = safe_get(item, ["source", "file", "doc"]) or "?"
        print(f"[{rank}] score={float(s):.4f}  idx={int(i)}  fonte={src}")
        print(textwrap.shorten(text.replace("\n", " "), width=SHOW_TEXT_CHARS, placeholder=" ..."))
        print()

    # Visualização 2D (PCA) opcional
    viz = input("Quer visualizar em 2D com PCA? (s/n): ").strip().lower()
    if viz == "s":
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2, random_state=42)
            X2 = pca.fit_transform(E_norm)

            plt.figure()
            plt.scatter(X2[:, 0], X2[:, 1], s=8)
            plt.title("Embeddings (PCA 2D)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.show()
        except Exception as e:
            print("Falha ao plotar (verifique se matplotlib e sklearn estão instalados). Erro:", e)

if __name__ == "__main__":
    main()
