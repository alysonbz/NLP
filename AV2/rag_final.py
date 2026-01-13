from utils.data_loader import load_jsonl_folder
from utils.vector_index import build_vector_store
from utils.rag_pipeline import rag_query
import pandas as pd
from openpyxl import Workbook
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# garantir que tokenizador existe
nltk.download("punkt")

dados_limpos_path = Path("AV2/dadoslimpos")
resultados_path = Path("AV2/resultados")
resultados_path.mkdir(exist_ok=True, parents=True)

ru_df = pd.read_csv(dados_limpos_path / "respostas_ru.csv")
biblioteca_df = pd.read_csv(dados_limpos_path / "respostas_b.csv")
respostas_turmas = pd.read_csv(dados_limpos_path / "respostas_turmas.csv")


perguntas_ru = ru_df["Perguntas"].to_list()
esperadas_ru = ru_df["Resposta esperada"].to_list()

perguntas_biblioteca = biblioteca_df["Perguntas"].to_list()
esperadas_biblioteca = biblioteca_df["Resposta esperada"].to_list()

perguntas_turmas = respostas_turmas["Perguntas"].to_list()
esperadas_turmas = respostas_turmas["Resposta esperada"].to_list()

respostas_ru = []
respostas_biblioteca = []
respostas_turmas = []

bleu_ru = []
bert_ru = []
bleu_biblioteca = []
bert_biblioteca = []
bleu_turmas = []
bert_turmas = []


def compute_bleu(expected, generated):
    """Calcula BLEU-1 / BLEU-2 simplificado para respostas curtas."""
    smoothie = SmoothingFunction().method1
    reference = expected.split()
    candidate = generated.split()

    if len(candidate) == 0:
        return 0.0

    bleu = sentence_bleu([reference], candidate, smoothing_function=smoothie)
    return float(bleu)


def compute_bertscore(expected, generated):
    """Calcula BERTScore (F1)."""
    P, R, F1 = bert_score([generated], [expected], lang="pt", verbose=False)
    return float(F1[0])


def main():
    # 1. Carregar documentos
    docs = load_jsonl_folder("AV2/dadoslimpos")

    # 2. Índice vetorial
    vectorstore = build_vector_store(docs, persist_directory="chroma_db")

    # 3. RAG + métricas para RU
    for pergunta, esperado in zip(perguntas_ru, esperadas_ru):
        resposta = rag_query(vectorstore, pergunta, "gpt-4o-mini")
        respostas_ru.append(resposta['answer'])

        bleu = compute_bleu(esperado, resposta['answer'])
        bert = compute_bertscore(esperado, resposta['answer'])

        bleu_ru.append(bleu)
        bert_ru.append(bert)

        print(f"[RU] Pergunta: {pergunta}")
        print(f"BLEU: {bleu:.4f} | BERTScore: {bert:.4f}")
        print("----------------------------------")
    
    # 4. RAG + métricas para BIBLIOTECA
    for pergunta, esperado in zip(perguntas_biblioteca, esperadas_biblioteca):
        resposta = rag_query(vectorstore, pergunta, "gpt-4o-mini")
        respostas_biblioteca.append(resposta['answer'])

        bleu = compute_bleu(esperado, resposta['answer'])
        bert = compute_bertscore(esperado, resposta['answer'])

        bleu_biblioteca.append(bleu)
        bert_biblioteca.append(bert)

        print(f"[BIBLIOTECA] Pergunta: {pergunta}")
        print(f"BLEU: {bleu:.4f} | BERTScore: {bert:.4f}")
        print("----------------------------------")

    # 5. RAG + métricas para TURMAS
    for pergunta, esperado in zip(perguntas_turmas, esperadas_turmas):
        resposta = rag_query(vectorstore, pergunta, "gpt-4o-mini")
        respostas_turmas.append(resposta['answer'])

        bleu = compute_bleu(esperado, resposta['answer'])
        bert = compute_bertscore(esperado, resposta['answer'])

        bleu_turmas.append(bleu)
        bert_turmas.append(bert)

        print(f"[TURMAS] Pergunta: {pergunta}")
        print(f"BLEU: {bleu:.4f} | BERTScore: {bert:.4f}")
        print("----------------------------------")



    # 6. Salvar no Excel
    salvar_excel_final()

def salvar_excel_final():

    df_ru = pd.DataFrame({
        "Perguntas": perguntas_ru,
        "Resposta esperada": esperadas_ru,
        "Resposta LLM": respostas_ru
    })

    df_biblioteca = pd.DataFrame({
        "Perguntas": perguntas_biblioteca,
        "Resposta esperada": esperadas_biblioteca,
        "Resposta LLM": respostas_biblioteca
    })

    df_turmas = pd.DataFrame({
        "Perguntas": perguntas_turmas,
        "Resposta esperada": esperadas_turmas,
        "Resposta LLM": respostas_turmas
    })

    # Novo: DataFrame somente de MÉTRICAS
    df_metrics = pd.DataFrame({
        "Pergunta": perguntas_ru + perguntas_biblioteca + perguntas_turmas,
        "Resposta esperada": esperadas_ru + esperadas_biblioteca + esperadas_turmas,
        "Resposta LLM":  respostas_ru + respostas_biblioteca + respostas_turmas,
        "BLEU": bleu_ru + bleu_biblioteca + bleu_turmas,
        "BERTScore": bert_ru + bert_biblioteca + bert_turmas,
        "Origem": ["ru"] * len(perguntas_ru) + ["biblioteca"] * len(perguntas_biblioteca) + ["turmas"] * len(perguntas_turmas)
    })

    # Escrever tudo no Excel
    with pd.ExcelWriter(resultados_path / "resultado_rag.xlsx", engine="openpyxl") as writer:
        df_ru.to_excel(writer, sheet_name="ru", index=False)
        df_biblioteca.to_excel(writer, sheet_name="biblioteca", index=False)
        df_turmas.to_excel(writer, sheet_name="turmas", index=False)
        df_metrics.to_excel(writer, sheet_name="metricas", index=False)

    print("Excel salvo como resultado_rag.xlsx")

if __name__ == "__main__":
    main()