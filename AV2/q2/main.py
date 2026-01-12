from utils.data_loader import load_txt_folder  # novo loader
from utils.vector_index import build_vector_store
from utils.rag_pipeline import rag_query

import pandas as pd
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk

# garantir tokenizador
nltk.download("punkt")

# ======== Configuração de paths ========
DADOS_PATH = Path("dados")  # pasta com os txt
XLSX_PATH = Path("perguntas_rag.xlsx")  # arquivo com perguntas e respostas
RESULTADOS_PATH = Path("resultados")
RESULTADOS_PATH.mkdir(exist_ok=True, parents=True)

# ======== Funções de Métrica ========
def compute_bleu(expected: str, generated: str):
    smoothie = SmoothingFunction().method1
    ref = expected.split()
    cand = generated.split()
    if len(cand) == 0:
        return 0.0
    return float(sentence_bleu([ref], cand, smoothing_function=smoothie))


def compute_bert(expected: str, generated: str):
    P, R, F1 = bert_score([generated], [expected], lang="pt", verbose=False)
    return float(F1[0])


def main():
    print("== 1. Carregando documentos RAG ==")
    docs = load_txt_folder(str(DADOS_PATH))

    print(f"Total de documentos carregados: {len(docs)}")

    print("== 2. Construindo índice vetorial ==")
    vectorstore = build_vector_store(
        docs,
        embedding_type="google",
        chunk_size=800,
        chunk_overlap=200,
        persist_directory="chroma_db",
        collection_name="times_football"
    )

    print("== 3. Lendo Excel de perguntas ==")
    df = pd.read_excel(XLSX_PATH)

    perguntas = df["perguntas"].astype(str).tolist()
    esperadas = df["resposta_esperada"].astype(str).tolist()

    respostas_llm = []
    metric_bleu = []
    metric_bert = []

    print("== 4. Executando RAG + Avaliação ==")
    for p, e in zip(perguntas, esperadas):
        res = rag_query(vectorstore, p, "gpt-4o-mini", temperature=0.4, k=3)
        answer = res["answer"]

        respostas_llm.append(answer)

        bleu = compute_bleu(e, answer)
        bert = compute_bert(e, answer)

        metric_bleu.append(bleu)
        metric_bert.append(bert)

        print(f"Pergunta: {p}")
        print(f"> Resposta LLM: {answer}")
        print(f"> BLEU: {bleu:.4f}  |  BERTScore: {bert:.4f}")
        print("-" * 50)

    print("== 5. Salvando Excel final ==")
    df_out = pd.DataFrame({
        "pergunta": perguntas,
        "resposta_esperada": esperadas,
        "resposta_llm": respostas_llm,
        "BLEU": metric_bleu,
        "BERTScore": metric_bert
    })

    df_out.to_excel(RESULTADOS_PATH / "resultado_rag.xlsx", index=False)
    print("Arquivo salvo em resultados/resultado_rag.xlsx")


if __name__ == "__main__":
    main()
