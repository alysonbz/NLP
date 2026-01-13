import pandas as pd
import json
from pathlib import Path

def salvar_jsonl(docs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

# ============================================================
# RESTAURANTE UNIVERSITÁRIO
# ============================================================

def processar_ru(csv_path, output_jsonl):
    df = pd.read_csv(csv_path)

    docs = []
    for idx, row in df.iterrows():

        texto = (
            f"Categoria: Restaurante Universitário\n"
            f"Serviço: {row.get('Serviço','')}\n"
            f"Dias: {row.get('Dias','')}\n"
            f"Horário: {row.get('Horário','')}\n"
            f"Local: {row.get('Local','')}"
        )

        docs.append({"id": idx, "text": texto})

    salvar_jsonl(docs, output_jsonl)


# ============================================================
# TURMAS UFC 2025.2
# ============================================================

def processar_turmas(csv_path, output_jsonl):
    df = pd.read_csv(csv_path)

    docs = []
    for idx, row in df.iterrows():

        texto = (
            f"Categoria: Turma\n"
            f"Disciplina: {row.get('Disciplina','')}\n"
            f"Turma: {row.get('Turma','')}\n"
            f"Docente: {row.get('Docente','')}\n"
            f"Tipo: {row.get('Tipo','')}\n"
            f"Situação: {row.get('Situação','')}\n"
            f"Horário: {row.get('Horário','')}\n"
            f"Local: {row.get('Local','')}"
        )

        docs.append({"id": idx, "text": texto})

    salvar_jsonl(docs, output_jsonl)


# ============================================================
# BIBLIOTECA UFC ITAPAJÉ
# ============================================================

def processar_biblioteca(csv_path, output_jsonl):
    df = pd.read_csv(csv_path)

    docs = []
    for idx, row in df.iterrows():

        texto = (
            f"Categoria: Biblioteca\n"
            f"Período: {row.get('Período','')}\n"
            f"Dias: {row.get('Dias','')}\n"
            f"Horário: {row.get('Horário','')}"
        )

        docs.append({"id": idx, "text": texto})

    salvar_jsonl(docs, output_jsonl)


# ============================================================
# EXECUTAR PARA OS 3 ARQUIVOS
# ============================================================

if __name__ == "__main__":
    base = Path("AV2/dadosav2")
    saida = Path("AV2/dadoslimpos")
    saida.mkdir(exist_ok=True, parents=True)

    processar_ru(
        base / "horarios_restaurante_ufc_itapaje.csv",
        saida / "ru_normalizado.jsonl"
    )

    processar_turmas(
        base / "turmas_ufc_2025_2.csv",
        saida / "turmas_normalizado.jsonl"
    )

    processar_biblioteca(
        base / "horarios_biblioteca_ufc.csv",
        saida / "biblioteca_normalizado.jsonl"
    )

    print("JSONL dos três arquivos gerados com sucesso!")
