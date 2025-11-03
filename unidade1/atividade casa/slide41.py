#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projeto: Extração e Limpeza de Dados com Expressões Regulares
Implementação completa até a Etapa 5.

Etapa 1 — Carregamento e exploração
Etapa 2 — Criação de padrões regex
Etapa 3 — Normalização
Etapa 4 — Validação
Etapa 5 — Relatório final
"""

from pathlib import Path
import re
import pandas as pd

# Caminhos
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR.parent / "dataset" / "dataset_sintetico_regex_ruidos.csv"
OUT_ETAPA1 = SCRIPT_DIR / "etapa1_relatorio_exploracao.csv"
OUT_ETAPA3 = SCRIPT_DIR / "etapa3_normalizado.csv"
OUT_ETAPA4 = SCRIPT_DIR / "etapa4_validacao.csv"
OUT_ETAPA5 = SCRIPT_DIR / "etapa5_relatorio.txt"

# Funções auxiliares e padrões (iguais às etapas anteriores)
def only_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")

def normalize_colname(c: str) -> str:
    return re.sub(r"[\s_\-]+", "", c.strip().lower())

# Padrões
NOME_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'´`^~\- ]{2,}$")
CPF_FMT_RE = re.compile(r"^\d{3}\.\d{3}\.\d{3}-\d{2}$")
CPF_DIG_RE = re.compile(r"^\d{11}$")
CNPJ_FMT_RE = re.compile(r"^\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}$")
CNPJ_DIG_RE = re.compile(r"^\d{14}$")
EMAIL_RE = re.compile(r"(?i)^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$")
FONE_ANY_RE = re.compile(r"(?:(?:\+?55)?\s*\(?\d{2}\)?[\s\-\.]*\d{4,5}[\s\-\.]*\d{4})")
CEP_RE = re.compile(r"^\d{5}-?\d{3}$")
DATA_DDMMAAAA_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
VALOR_BR_RE = re.compile(r"^(?:R\$\s*)?(?:\d{1,3}(?:\.\d{3})*|\d+),\d{2}$")
URL_RE = re.compile(r"(?i)^(https?://)?([A-Z0-9.-]+\.[A-Z]{2,})(/[^\s]*)?$")

# ------------------------------------------------------------
# Validação CPF/CNPJ
# ------------------------------------------------------------
def valida_cpf(d: str) -> bool:
    d = only_digits(d)
    if len(d) != 11 or d == d[0] * 11:
        return False
    soma = sum(int(d[i]) * (10 - i) for i in range(9))
    dv1 = (soma * 10) % 11
    dv1 = 0 if dv1 == 10 else dv1
    soma = sum(int(d[i]) * (11 - i) for i in range(10))
    dv2 = (soma * 10) % 11
    dv2 = 0 if dv2 == 10 else dv2
    return d[-2:] == f"{dv1}{dv2}"

def valida_cnpj(d: str) -> bool:
    d = only_digits(d)
    if len(d) != 14 or d == d[0] * 14:
        return False
    pesos1 = [5,4,3,2,9,8,7,6,5,4,3,2]
    soma1 = sum(int(d[i]) * pesos1[i] for i in range(12))
    dv1 = 11 - (soma1 % 11)
    dv1 = 0 if dv1 >= 10 else dv1
    pesos2 = [6] + pesos1
    soma2 = sum(int(d[i]) * pesos2[i] for i in range(13))
    dv2 = 11 - (soma2 % 11)
    dv2 = 0 if dv2 >= 10 else dv2
    return d[-2:] == f"{dv1}{dv2}"

# Normalização

def normaliza_cpf(s: str):
    d = only_digits(s)
    if len(d) != 11:
        return None
    return f"{d[0:3]}.{d[3:6]}.{d[6:9]}-{d[9:11]}"

def normaliza_cnpj(s: str):
    d = only_digits(s)
    if len(d) != 14:
        return None
    return f"{d[0:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:14]}"

def normaliza_fone(s: str):
    d = only_digits(s)
    if d.startswith("55") and len(d) >= 12:
        d = d[2:]
    d = d.lstrip("0")
    if len(d) == 11:
        ddd, p1, p2 = d[:2], d[2:7], d[7:]
        return f"({ddd}) {p1}-{p2}"
    return None

# Etapa 1 a 4

def executar_etapas_1_a_4():
    if OUT_ETAPA4.exists():
        print("[✓] Etapas 1–4 já executadas — prosseguindo para a Etapa 5.")
        return pd.read_csv(OUT_ETAPA4)
    raise FileNotFoundError("Execute as Etapas 1–4 antes da Etapa 5.")

# ETAPA 5 elatório final

def etapa5_relatorio(df: pd.DataFrame):
    print("\n=== Etapa 5 — Relatório Final ===")
    total = len(df)

    # ---- Quantidade e percentual de válidos/inválidos ----
    if "cpf__valido" in df.columns and "cnpj__valido" in df.columns:
        total_cpf_validos = df["cpf__valido"].sum()
        total_cnpj_validos = df["cnpj__valido"].sum()
        perc_cpf_validos = total_cpf_validos / total * 100
        perc_cnpj_validos = total_cnpj_validos / total * 100
    else:
        total_cpf_validos = total_cnpj_validos = perc_cpf_validos = perc_cnpj_validos = 0

    # ---- Campos com mais inconsistências ----
    status_cols = [c for c in df.columns if c.endswith("__status")]
    inconsistencias = []
    for col in status_cols:
        cont = df[col].value_counts()
        inconsistencias.append({
            "coluna": col.replace("__status", ""),
            "incompletos": cont.get("incompleto", 0),
            "ruidos": cont.get("ruído/ inválido", 0)
        })
    inc_df = pd.DataFrame(inconsistencias)
    inc_df["total_inconsistencias"] = inc_df["incompletos"] + inc_df["ruidos"]
    top_incos = inc_df.sort_values("total_inconsistencias", ascending=False).head(3)

    # ---- Exemplo antes/depois ----
    exemplos = []
    if "cpf" in df.columns and "cpf_normalizado" in df.columns:
        exemplos.append(("CPF", df.loc[df["cpf_normalizado"].notna(), ["cpf","cpf_normalizado"]].head(3)))
    if "cnpj" in df.columns and "cnpj_normalizado" in df.columns:
        exemplos.append(("CNPJ", df.loc[df["cnpj_normalizado"].notna(), ["cnpj","cnpj_normalizado"]].head(3)))
    if "telefone" in df.columns and "telefone_normalizado" in df.columns:
        exemplos.append(("Telefone", df.loc[df["telefone_normalizado"].notna(), ["telefone","telefone_normalizado"]].head(3)))

    # ---- Metodologia de correção ----
    metodologia = """\nMetodologia de Correção:
1. Identificação de padrões usando expressões regulares específicas para cada tipo de dado.
2. Remoção de caracteres não numéricos em CPFs, CNPJs e telefones.
3. Normalização para formatos padronizados:
   - CPF → XXX.XXX.XXX-YY
   - CNPJ → XX.XXX.XXX/0001-YY
   - Telefone → (XX) 9XXXX-XXXX
4. Validação dos dígitos verificadores de CPF e CNPJ.
5. Classificação dos registros em válidos, incompletos e ruidosos.
"""

    # relatório
    linhas = []
    linhas.append("RELATÓRIO FINAL — ETAPA 5\n")
    linhas.append(f"Número total de registros: {total}\n")
    linhas.append(f"CPF válidos: {total_cpf_validos} ({perc_cpf_validos:.1f}%)\n")
    linhas.append(f"CNPJ válidos: {total_cnpj_validos} ({perc_cnpj_validos:.1f}%)\n")
    linhas.append("\nCampos com mais inconsistências:\n")
    linhas.append(top_incos.to_string(index=False))
    linhas.append("\n\nExemplos antes/depois da limpeza:\n")
    for nome, ex in exemplos:
        linhas.append(f"\n{nome}:\n{ex.to_string(index=False)}\n")
    linhas.append(metodologia)

    relatorio_texto = "\n".join(linhas)

    #exibe e salva
    print(relatorio_texto)
    with open(OUT_ETAPA5, "w", encoding="utf-8") as f:
        f.write(relatorio_texto)
    print(f"\n[OK] Relatório salvo em: {OUT_ETAPA5}")

# ------------------------------------------------------------
# Execução principal
# ------------------------------------------------------------
def main():
    df = executar_etapas_1_a_4()
    etapa5_relatorio(df)

if __name__ == "__main__":
    main()
