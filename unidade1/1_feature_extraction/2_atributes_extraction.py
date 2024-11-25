import re
import pandas as pd


# Atividade
# 1) Um função que retorne a quantidade de sentenças em um texto.
def contar_sentencas(text):
    result = re.findall(r'[^.!?]+[.!?]', text)
    return result
texto1 = "Eu sou o cara."
print(contar_sentencas(texto1))


# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
def contar_palavras_maisculas(text):
    result = len(re.findall(r"[A-Z][a-z]*\b", text))
    return result
texto2 = "Eu não GOSTO do João"
print(contar_palavras_maisculas(texto2))


# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.
def count_numbers(text):
    result = len(re.findall("\d", text))
    return result
texto3 = "Eu tenho 10 anos e 10 meses"
print(count_numbers(texto3))


# 4) Uma função que retorne a quantidade de palavras que estão  em caixa alta.
def contar_caixa_alta(text):
    result = re.findall(r'\b[A-Z]+\b', text)
    return result
texto4 = "Eu ODEIO ODEIO ele"
print(contar_caixa_alta(texto4))

# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.
df = pd.DataFrame({"text": [texto1, texto2, texto3, texto4]})
print(df)

# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.
df_resultados = pd.DataFrame({
    "Sentenças": df["text"].apply(contar_sentencas),
    "Palavras com maiúscula": df["text"].apply(contar_palavras_maisculas),
    "Números": df["text"].apply(count_numbers),
    "Palavras em caixa alta": df["text"].apply(contar_caixa_alta)
})

print(df_resultados)