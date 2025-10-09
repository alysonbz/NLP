# -*- coding: utf-8 -*-
"""
Q2 e Q3 – PLN em PT com spaCy + NLTK + (opcional) LanguageTool
- 2) Tabela com significados das POS que aparecem no texto
- 3) (a) Correção de texto (LanguageTool) com fallback sem Java
     (b) Tokenização + POS
     (c) Remoção de stopwords
     (d) Tabela token / stem / lemma (com e sem stopwords)
Gera ainda 3 CSVs: tabela_tokens_com_stopwords.csv,
                   tabela_tokens_sem_stopwords.csv,
                   tabela_significados_pos.csv
"""

import sys
print("Executando com:", sys.executable)

# -------------------- Imports --------------------
import pandas as pd
import spacy
from spacy.util import is_package
from spacy.cli import download as spacy_download

# NLTK: stopwords + stemmer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# -------------------- Recursos / downloads únicos --------------------
# NLTK stopwords
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# spaCy PT
PT_MODEL = "pt_core_news_sm"
if not is_package(PT_MODEL):
    spacy_download(PT_MODEL)
nlp = spacy.load(PT_MODEL)

# Tenta habilitar LanguageTool (requer Java). Se não houver, seguimos sem correção.
tool = None
try:
    import language_tool_python
    tool = language_tool_python.LanguageTool("pt-BR")
    print("[INFO] LanguageTool ativo (Java detectado).")
except Exception as e:
    print("[AVISO] LanguageTool indisponível (sem Java?). Usando texto original, sem correção.")
    tool = None

# NLTK recursos
stemmer = SnowballStemmer("portuguese")
stop_pt = set(stopwords.words("portuguese"))

# -------------------- Mapa de POS (UPOS → PT) --------------------
UPOS_MEANINGS = {
    "ADJ": "Adjetivo",
    "ADP": "Adposição (preposição)",
    "ADV": "Advérbio",
    "AUX": "Verbo auxiliar",
    "CCONJ": "Conjunção coordenativa",
    "DET": "Determinante",
    "INTJ": "Interjeição",
    "NOUN": "Substantivo comum",
    "NUM": "Numeral",
    "PART": "Partícula",
    "PRON": "Pronome",
    "PROPN": "Substantivo próprio",
    "PUNCT": "Pontuação",
    "SCONJ": "Conjunção subordinativa",
    "SYM": "Símbolo",
    "VERB": "Verbo",
    "X": "Desconhecido"
}

# -------------------- Funções --------------------
def corrigir_texto(texto: str) -> str:
    """Correção com LanguageTool; se indisponível, devolve o original."""
    if tool is None:
        return texto

    # 1) Corrige por frase (melhora alguns casos)
    sents = [texto[s.start_char:s.end_char] for s in nlp(texto).sents]
    corrigidas = []
    for s in sents:
        try:
            corrigidas.append(tool.correct(s))
        except Exception:
            corrigidas.append(s)
    texto1 = " ".join(corrigidas)

    # 2) Segunda passada no texto inteiro (pega sobras)
    try:
        texto2 = tool.correct(texto1)
    except Exception:
        texto2 = texto1
    return texto2

def pos_ajustes(texto: str) -> str:
    """Ajustes pós-correção (opcional) para casos frequentes em PT."""
    rep = {
        " Nos ": " Nós ",
        " nos ": " nós ",
        " faze ": " fazer ",
        " amanha ": " amanhã ",
    }
    t = f" {texto} "
    for k, v in rep.items():
        t = t.replace(k, f" {v} ")
    return t.strip()

def tabela_pos_significados(texto: str) -> pd.DataFrame:
    """Q2: POS distintas do texto + significado em PT."""
    doc = nlp(texto)
    tags = sorted({t.pos_ for t in doc if not t.is_space})
    rows = [{"POS": tag, "Significado (PT)": UPOS_MEANINGS.get(tag, "(sem mapeamento)")} for tag in tags]
    return pd.DataFrame(rows)

def analisar_texto(texto_corrigido: str):
    """Q3b/c/d: Tokeniza, marca stopwords, gera token/stem/lemma e POS."""
    doc = nlp(texto_corrigido)
    rows = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        rows.append({
            "token": tok.text,
            "stem": stemmer.stem(tok.text.lower()),
            "lemma": tok.lemma_,
            "is_stopword": tok.text.lower() in stop_pt,
            "POS": tok.pos_
        })
    df = pd.DataFrame(rows)
    df_sem_sw = df[~df["is_stopword"]].reset_index(drop=True)
    return doc, df, df_sem_sw

# -------------------- Execução (exemplo) --------------------
if __name__ == "__main__":
    # Texto com erros gramaticais/ortográficos (troque pelo que o professor pedir)
    texto_errado = (
        "Eu foi na escola ontem e encontrei as menina estudando portugues. "
        "Nos vai faze a prova amanha cedo, ta bom?"
    )

    # 3a) Correção (com fallback) + pós-ajuste leve
    texto_corrigido = corrigir_texto(texto_errado)
    texto_corrigido = pos_ajustes(texto_corrigido)

    print("\n=== 3a) Texto corrigido ===\n", texto_corrigido)

    # 3b/3c/3d) Tokenização, stopwords e tabela
    doc, df_completo, df_sem_sw = analisar_texto(texto_corrigido)

    print("\n=== 3b) Tokens e POS (após correção) ===")
    print([(t.text, t.pos_) for t in doc if not t.is_space])

    print("\n=== 3c) Contagem ===")
    print(f"Tokens (sem pontuação): {len(df_completo)}")
    print(f"Tokens sem stopwords:   {len(df_sem_sw)}")

    print("\n=== 3d) Tabela Token / Stem / Lemma (COM stopwords) ===")
    print(df_completo[["token", "stem", "lemma"]].to_string(index=False))

    print("\n=== 3d) Tabela Token / Stem / Lemma (SEM stopwords) ===")
    print(df_sem_sw[["token", "stem", "lemma"]].to_string(index=False))

    # 2) POS + significados
    print("\n=== 2) Significados das POS presentes no texto ===")
    df_pos = tabela_pos_significados(texto_corrigido)
    print(df_pos.to_string(index=False))

    # Salva CSVs para anexar
    df_completo.to_csv("tabela_tokens_com_stopwords.csv", index=False, encoding="utf-8")
    df_sem_sw.to_csv("tabela_tokens_sem_stopwords.csv", index=False, encoding="utf-8")
    df_pos.to_csv("tabela_significados_pos.csv", index=False, encoding="utf-8")
    print("\nArquivos salvos: tabela_tokens_com_stopwords.csv, tabela_tokens_sem_stopwords.csv, tabela_significados_pos.csv")

    # encerra LT se aberto
    try:
        if tool is not None:
            tool.close()
    except Exception:
        pass
