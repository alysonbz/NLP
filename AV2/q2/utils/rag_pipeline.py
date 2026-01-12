from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import time

# =========================
# SPARSE RETRIEVER
# =========================
def build_sparse_retriever(docs):
    return BM25Retriever.from_documents(docs)

# =========================
# RRF FUSION (Reciprocal Rank Fusion)
# =========================
def rrf_fusion(dense_docs, sparse_docs, k=5):
    scores = {}

    def add_scores(docs, w=1.0):
        for rank, d in enumerate(docs):
            key = d.page_content
            scores[key] = scores.get(key, 0) + w * 1.0 / (60 + rank)

    add_scores(dense_docs, w=1.0)
    add_scores(sparse_docs, w=1.0)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [d for d, _ in sorted_docs][:k]
    return selected


# =========================
# RAG QUERY (com modo)
# =========================
def rag_query(
    vectorstore,
    query: str,
    llm_provider: str = "openai",
    llm_model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    k: int = 2,
    retrieval_mode: str = "dense",   # "dense", "sparse", "hybrid"
    sparse_retriever=None
):
    """
    llm_provider: "openai" ou "google"
    retrieval_mode: "dense", "sparse", "hybrid"
    """
    t0 = time.perf_counter()

    retrieval_mode = retrieval_mode.lower()

    # =========================
    # 1. Recuperação
    # =========================
    if retrieval_mode == "dense":
        docs = vectorstore.similarity_search(query, k=k)

    elif retrieval_mode == "sparse":
        if sparse_retriever is None:
            raise ValueError("retrieval_mode='sparse' requer sparse_retriever")
        docs = sparse_retriever.invoke(query)

    elif retrieval_mode == "hybrid":
        if sparse_retriever is None:
            raise ValueError("retrieval_mode='hybrid' requer sparse_retriever")
        dense_docs = vectorstore.similarity_search(query, k=k)
        sparse_docs = sparse_retriever.invoke(query)
        docs = rrf_fusion(dense_docs, sparse_docs, k=k)

    else:
        raise ValueError(f"retrieval_mode '{retrieval_mode}' inválido.")

    retrieval_time = time.perf_counter() - t0

    # =========================
    # 2. Montar o contexto
    # =========================
    if not isinstance(docs, list):
        docs = [docs]

    docs = [
        d if isinstance(d, Document) else Document(page_content=str(d))
        for d in docs
    ]

    context = "\n\n".join([d.page_content for d in docs])

    system_prompt = (
            "Você é um assistente especializado em clubes de futebol do Brasil. "
            "Responda apenas com base no contexto fornecido. "
            "Sua resposta deve ser objetiva e curta, sem adicionar informações não presentes no contexto."
            "\n\n=== CONTEXTO ===\n"
            + context
    )
    # =========================
    # 3. Selecionar LLM
    # =========================
    t1 = time.perf_counter()

    provider = llm_provider.lower()

    if provider == "openai":
        llm = ChatOpenAI(
            model_name=llm_model_name,
            temperature=temperature
        )

    elif provider == "google":
        llm = ChatGoogleGenerativeAI(
            model=llm_model_name,
            temperature=temperature
        )

    else:
        raise ValueError(f"provider '{llm_provider}' não suportado.")

    # =========================
    # 4. Gerar resposta
    # =========================
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])

    generation_time = time.perf_counter() - t1
    total_time = time.perf_counter() - t0

    return {
        "answer": response.content,
        "context_docs": docs,
        "retrieval_mode": retrieval_mode,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time
    }
