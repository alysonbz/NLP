# 2_atributes_extraction.py
import re
import pandas as pd


# -------------------------
# 1) quantidade de sentenças
# -------------------------
def count_sentences(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    # divide por ., ?, ! e ignora pedaços vazios
    partes = re.split(r"[\.!?]+", text)
    return sum(1 for p in partes if p.strip())


# -------------------------
# 2) palavras que começam com maiúscula
# -------------------------
def count_capitalized_words(text: str) -> int:
    if not isinstance(text, str):
        return 0
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", text)
    total = 0
    for tok in tokens:
        if tok[0].isalpha() and tok[0].isupper():
            total += 1
    return total


# -------------------------
# 3) quantidade de caracteres numéricos
# -------------------------
def count_numeric_chars(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return sum(ch.isdigit() for ch in text)


# -------------------------
# 4) palavras em CAIXA ALTA
# -------------------------
def count_all_uppercase_words(text: str) -> int:
    if not isinstance(text, str):
        return 0
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)
    return sum(tok.isupper() for tok in tokens)


if __name__ == "__main__":
    # 5) cria o dataframe de 1 coluna e 4 linhas com frases de teste
    df_textos = pd.DataFrame({
        "texto": [
            "Maria visitou Paris. Foi incrível!",
            "Em 2024, JOÃO correu 10km? Sim!",
            "A NASA lançou um FOGUETE em 1998!!!",
            "Hoje está nublado; Amanhã, Sol forte."
        ]
    })

    print("DataFrame com os textos (4 linhas, 1 coluna):")
    print(df_textos, "\n")

    # 6) cria o dataframe de 4x4 com os resultados das funções 1–4
    df_atributos = pd.DataFrame({
        "sentencas":       df_textos["texto"].apply(count_sentences),
        "palavras_maiusc": df_textos["texto"].apply(count_capitalized_words),
        "numericos":       df_textos["texto"].apply(count_numeric_chars),
        "caixa_alta":      df_textos["texto"].apply(count_all_uppercase_words),
    })

    print("DataFrame final (4x4) com os atributos extraídos:")
    print(df_atributos)
