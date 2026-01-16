import os
import pandas as pd
import chromadb

CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", ".chroma_db")
COLLECTION_NAME = os.getenv("RAG_CHROMA_COLLECTION", "rag_docs")

client = chromadb.PersistentClient(path=CHROMA_DIR)
col = client.get_collection(COLLECTION_NAME)

total = col.count()
print("\nChromaDB (dump all)")
print(f"Chroma dir: {CHROMA_DIR}")
print(f"Collection: {COLLECTION_NAME}")
print(f"Total na coleção (count): {total}")

# Tamanho do lote
BATCH_SIZE = int(os.getenv("RAG_CHROMA_BATCH", "1000"))

rows = []
offset = 0

include = ["documents", "metadatas", "embeddings"]

while offset < total:
    batch = col.get(
        limit=BATCH_SIZE,
        offset=offset,
        include=include,
    )

    ids = batch.get("ids") or []
    docs = batch.get("documents") or []
    metas = batch.get("metadatas") or []
    embs = batch.get("embeddings")

    n = len(ids)

    for i in range(n):
        meta = metas[i] if i < len(metas) and metas[i] is not None else {}

        emb_i = None
        if embs is not None and i < len(embs):
            emb_i = embs[i]

        if emb_i is None:
            emb_list = []
            dim = 0
            emb_0_8 = []
        else:
            emb_list = emb_i.tolist() if hasattr(emb_i, "tolist") else list(emb_i)
            dim = len(emb_list)
            emb_0_8 = emb_list[:8]

        text = docs[i] if i < len(docs) and docs[i] is not None else ""
        rows.append({
            "id": ids[i],
            "doc": meta.get("doc"),
            "chunk_id": meta.get("chunk_id"),
            "dim": dim,
            "embedding_0_8": emb_0_8,
            "text_preview": (text.replace("\n", " ") + "...") if text else ""
        })

    offset += n
    print(f"Progresso: {offset}/{total}")

df = pd.DataFrame(rows)

print("\nAmostra (primeiras 10 linhas):")
print(df.head(10).to_string(index=False))

out = "outputs/chromadb_all.csv"
df.to_csv(out, index=False, encoding="utf-8")
print(f"\nSalvo: {out}")
