import json

RESULT_FILE = "outputs/q2_results.json"

with open(RESULT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\n===== RESULTADOS DO PIPELINE RAG (BASELINE) =====\n")

print(f"Retriever utilizado: {data.get('retriever')}")
print(f"LLM utilizada: {data.get('llm')}")
print(f"Nº de documentos: {data.get('n_docs')}")
print(f"Nº de chunks: {data.get('n_chunks')}")
print(f"Top-k: {data.get('top_k')}")
print("\n-----------------------------------------------\n")

for i, case in enumerate(data["cases"], 1):
    print(f"Pergunta {i}:")
    print(case["question"])

    # tenta descobrir automaticamente a chave dos documentos
    doc_key = None
    for k in case.keys():
        if "doc" in k.lower():
            doc_key = k
            break

    print("\nDocumentos recuperados:")
    if doc_key:
        for d in case[doc_key]:
            print(f" - {d}")
    else:
        print(" - (não encontrado no JSON)")

    print("\nResposta gerada:")
    print(case["answer"])

    print("\nAvaliação:", "CORRETA" if case["ok"] else "INCORRETA")
    print("-" * 70)

print("\nAccuracy proxy final:", data["accuracy_proxy"])
