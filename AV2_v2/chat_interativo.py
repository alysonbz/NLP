# pip install chromadb numpy requests

#   baixar um LLM e um modelo de embeddings:
#        ollama pull llama3.2
#        ollama pull bge-m3    (ou: nomic-embed-text / mxbai-embed-large)



import os
import time
import json
import hashlib
import numpy as np
import requests
import chromadb



# Config (via env, sem argparse)
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
DOCS_DIR = os.getenv("RAG_DOCS_DIR", "rag_docs")


CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))


# ChromaDB (persistência)
CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", ".chroma_db")
COLLECTION_NAME = os.getenv("RAG_CHROMA_COLLECTION", "rag_docs")
REINDEX_ON_CHANGE = os.getenv("RAG_REINDEX_ON_CHANGE", "1").strip() not in ("0", "false", "False")

BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH", "32"))

SYSTEM_RULE = "Regra obrigatória: use apenas o contexto e recuse quando faltar evidência."



# Utilitários Ollama
def ollama_is_up():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False



def ollama_generate(system_prompt: str, prompt: str, timeout=120) -> str:
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")



def ollama_embed(texts, timeout=120) -> np.ndarray:
    """
    texts: list[str]
    retorna: np.ndarray shape (n, dim), float32
    """
    url = f"{OLLAMA_BASE}/api/embed"
    payload = {"model": EMBED_MODEL, "input": texts}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    embs = data.get("embeddings")
    if embs is None:
        raise ValueError(f"Resposta inesperada do Ollama: chaves={list(data.keys())}")
    return np.asarray(embs, dtype=np.float32)



def load_txt_docs(folder):
    docs = {}
    if not os.path.isdir(folder):
        return docs
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(".txt"):
            path = os.path.join(folder, fn)
            with open(path, "r", encoding="utf-8") as f:
                docs[fn] = f.read().strip()
    return docs



def chunk_text(text, chunk_size=900, overlap=120):
    text = " ".join((text or "").split())
    chunks = []
    i = 0
    stride = max(1, chunk_size - overlap)
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += stride
    return chunks



def fingerprint_corpus(docs: dict) -> str:
    """
    Hash do corpus + parâmetros relevantes.
    Se mudar qualquer texto/arquivo/param, muda o fingerprint -> reindexa.
    """
    h = hashlib.sha256()
    h.update(f"CHUNK_SIZE={CHUNK_SIZE};OVERLAP={CHUNK_OVERLAP};EMBED_MODEL={EMBED_MODEL}".encode("utf-8"))
    for name in sorted(docs.keys()):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(docs[name].encode("utf-8", errors="ignore"))
        h.update(b"\0\0")
    return h.hexdigest()[:16]



# Prompt RAG
def build_prompt(question, retrieved):
    context_parts = []
    for r in retrieved:
        header = f"[{r['meta']['doc']} | chunk {r['meta']['chunk_id']} | score = {r['score']:.3f}]"
        context_parts.append(header + "\n" + r["text"])
    context = "\n\n".join(context_parts)

    lines = []
    lines.append("Você é um assistente que responde SOMENTE com base no CONTEXTO fornecido.")
    lines.append("Se a resposta não estiver no contexto, responda exatamente:")
    lines.append("\"Não há informação suficiente no contexto.\"")
    lines.append("")
    lines.append("CONTEXTO:")
    lines.append(context)
    lines.append("")
    lines.append("PERGUNTA:")
    lines.append(question)
    lines.append("")
    lines.append("Responda em português, direto e bem justificado. Se possível, cite doc/chunk.")
    return "\n".join(lines)



# Execução direta
docs = load_txt_docs(DOCS_DIR)
if not docs:
    print(f"[ERRO] Não encontrei arquivos .txt em '{DOCS_DIR}'.")
    print("Crie a pasta rag_docs e coloque seus .txt lá.")
    raise SystemExit(1)

print("Docs carregados:", len(docs))
print("Arquivos:", ", ".join(list(docs.keys())[:8]) + ("..." if len(docs) > 8 else ""))


if not ollama_is_up():
    print("[ERRO] Ollama não está acessível em:", OLLAMA_BASE)
    print("Abra o Ollama e rode: ollama pull llama3.2 (ou outro modelo) e tente novamente.")
    raise SystemExit(1)


# Monta corpus em chunks
chunks, metas = [], []
for doc_name, content in docs.items():
    for idx, ch in enumerate(chunk_text(content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)):
        chunks.append(ch)
        metas.append({"doc": doc_name, "chunk_id": idx})

print("Total de chunks:", len(chunks))


# Fingerprint para controle de reindex
fp = fingerprint_corpus(docs)


# ChromaDB: abre/cria coleção
client = chromadb.PersistentClient(path=CHROMA_DIR)
col = client.get_or_create_collection(name=COLLECTION_NAME)


# Guardamos um "estado" simples numa collection separada (ou arquivo).
state_path = os.path.join(CHROMA_DIR, f"{COLLECTION_NAME}__state.json")
os.makedirs(CHROMA_DIR, exist_ok=True)


prev_state = {}
if os.path.exists(state_path):
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            prev_state = json.load(f) or {}
    except Exception:
        prev_state = {}


needs_reindex = (
    prev_state.get("fingerprint") != fp
    or prev_state.get("embed_model") != EMBED_MODEL
    or prev_state.get("chunk_size") != CHUNK_SIZE
    or prev_state.get("chunk_overlap") != CHUNK_OVERLAP
)



# Indexação / Reindexação
def reset_collection():
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(name=COLLECTION_NAME)


if needs_reindex and REINDEX_ON_CHANGE:
    print(f"Corpus mudou (fingerprint={fp}). Reindexando coleção '{COLLECTION_NAME}'...")
    col = reset_collection()

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for start in range(0, len(chunks), BATCH_SIZE):
        batch_docs = chunks[start:start + BATCH_SIZE]
        batch_meta = metas[start:start + BATCH_SIZE]

        # IDs estáveis: doc::chunk_id::fingerprint
        batch_ids = [f"{m['doc']}::{m['chunk_id']}::{fp}" for m in batch_meta]

        batch_emb = ollama_embed(batch_docs)

        ids.extend(batch_ids)
        documents.extend(batch_docs)
        metadatas.extend(batch_meta)
        embeddings.extend(batch_emb.tolist())

        print(f"  - indexados {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    col.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    # salva estado
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fingerprint": fp,
                "num_chunks": len(chunks),
                "embed_model": EMBED_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "updated_at": time.time(),
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Reindex concluído. Persistido em:", CHROMA_DIR)
else:
    # Se não reindexar, assumimos que o índice existente é válido
    # (estado existente deve bater, ou o usuário desativou reindex)
    print(f"Usando coleção existente '{COLLECTION_NAME}' em {CHROMA_DIR}.")
    if needs_reindex and not REINDEX_ON_CHANGE:
        print("[AVISO] Corpus mudou, mas RAG_REINDEX_ON_CHANGE=0; resultados podem ficar inconsistentes.")



# Retrieval via Chroma
def retrieve(query, top_k=TOP_K):
    q_emb = ollama_embed([query])[0].tolist()

    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs_out = res.get("documents", [[]])[0]
    metas_out = res.get("metadatas", [[]])[0]
    dists_out = res.get("distances", [[]])[0]

    # Chroma pode retornar distance como "1 - cosine_similarity" dependendo da config interna.
    # Como estamos fornecendo embeddings, a métrica varia por versão/config.
    # Para exibição, convertemos para "score aproximado" quando fizer sentido:
    results = []
    for text, m, d in zip(docs_out, metas_out, dists_out):
        # Heurística: se d estiver em [0,2], pode ser cosine distance; score=1-d
        try:
            d = float(d)
            score = 1.0 - d
        except Exception:
            score = 0.0
        results.append({"score": score, "text": text, "meta": m})
    return results



print("\n" + "=" * 80)  # para fazer uma linha
print("RAG Chat (ChromaDB).")
print(f"LLM Ollama: {OLLAMA_MODEL} | Embeddings: {EMBED_MODEL} | Base: {OLLAMA_BASE} | Docs: {DOCS_DIR}")
print(f"Chroma: {CHROMA_DIR} | Collection: {COLLECTION_NAME}")
print(f"Chunk_size = {CHUNK_SIZE} chars | overlap = {CHUNK_OVERLAP} chars | top_k = {TOP_K}")
print("Para fechar o chat: sair / exit / quit")
print("=" * 80 + "\n")   # para fazer uma linha



while True:
    question = input("Você: ").strip()
    if not question:
        continue
    if question.lower() in ["sair", "exit", "quit"]:
        print("Encerrando.")
        break

    retrieved = retrieve(question, top_k=TOP_K)


    print("\nArquivos utilizados na busca:")
    for r in retrieved:
        print(f"- {r['meta'].get('doc')} | chunk {r['meta'].get('chunk_id')} | score~={r['score']:.3f}")

    prompt = build_prompt(question, retrieved)


    try:
        answer = ollama_generate(SYSTEM_RULE, prompt, timeout=300)
    except Exception as e:
        print("\nFalha ao chamar Ollama:", str(e))
        print("Verifique se o serviço está rodando e o modelo existe (ollama list).")
        continue


    print("\nLLM:", answer.strip(), "\n")
    time.sleep(0.05)
