from langchain_openai import ChatOpenAI

def rag_query(vectorstore, query: str, llm_model_name: str = "gpt-4o-mini", temperature: float = 0.0, k: int = 4):
    # 1. Recuperar documentos relevantes via Chroma
    docs = vectorstore.similarity_search(query, k=k)

    # 2. Construir contexto
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Criar prompt
    system_prompt = (
        "Você é um assistente que responde perguntas com base no contexto abaixo. "
        "Se você não souber a resposta, diga que não sabe.\n\n"
        "Contexto:\n" + context
    )

    # 4. Chamar o LLM
    llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])

    answer = response.content
    return {"answer": answer, "context_docs": docs}
