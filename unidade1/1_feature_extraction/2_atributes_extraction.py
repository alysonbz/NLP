# Atividade

import pandas as pd
import re

# 1) Uma função que retorne a quantidade de sentenças em um texto.
def count_sentences(text):
    # Conta divisões por ., ?, ! —
    return len([s for s in re.split(r'[.!?]+', text) if s.strip() != ""])

# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
def count_capitalized_words(text):
    words = text.split()
    return sum(1 for w in words if w[0].isupper())

# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.
def count_digits(text):
    return sum(ch.isdigit() for ch in text)

# 4) Uma função que retorne a quantidade de palavras que estão  em caixa alta.
def count_uppercase_words(text):
    words = text.split()
    return sum(1 for w in words if w.isupper())

# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.
data = {
    "textos": [
        "Olá pessoal! Hoje é 22/10/2025.",
        "O BTS vai fazer Turnê Mundial em 2026!!!",
        "Marilia ESTUDA Ciência de DADOS desde 2023.",
        "123 TESTE de TEXTO com NÚMEROS 456."
    ]
}
df = pd.DataFrame(data)

# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.
df_result = pd.DataFrame({
    "sentencas": df["textos"].apply(count_sentences),
    "palavras_Maiuscula": df["textos"].apply(count_capitalized_words),
    "digitos": df["textos"].apply(count_digits),
    "palavras_CAIXA_ALTA": df["textos"].apply(count_uppercase_words),
})

print(df)
print(df_result)

