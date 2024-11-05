import pandas as pd
import re

# 1) Um função que retorne a quantidade de sentenças em um texto
"I really like a subject"
def contar_sentenças(texto):
    sentencas = re.split(r'[.!?]+', texto)
    sentencas = [s for s in sentencas if s.strip()]
    return len(sentencas)


# Função 2: Quantidade de palavras que começam com letra maiúscula no texto.
def contar_palavras_maiusculas(texto):
    palavras = texto.split()
    return sum(1 for palavra in palavras if palavra[0].isupper())


# Função 3: Retorna a quantidade de caracteres numéricos no texto.
def contar_caracteres_numericos(texto):
    return sum(1 for char in texto if char.isdigit())

# Função 4: Quantidade de palavras em caixa alta.
def contar_palavras_caixa_alta(texto):
    palavras = texto.split()
    return sum(1 for palavra in palavras if palavra.isupper())

# Questão 5: 1 coluna e 4 linhas com textos para teste
textos = [
    "Brasil, Brasil!",
    "Simplemente o MAIORAL no seu continente!",
    "O maioral mesmo depois do 7x1 para a Alemanha",
    "Entao vamos exaltar nosso País Brasil na copa de 2026"
]
df_textos = pd.DataFrame(textos, columns=["Texto"])

# Questão 6: Aplicação das funções usando o método apply() para criar um DataFrame de 4x4 com os resultados
df_resultados = pd.DataFrame()
df_resultados['Quantidade de Sentenças'] = df_textos['Texto'].apply(contar_sentenças)
df_resultados['Palavras com Maiúscula'] = df_textos['Texto'].apply(contar_palavras_maiusculas)
df_resultados['Caracteres Numéricos'] = df_textos['Texto'].apply(contar_caracteres_numericos)
df_resultados['Palavras em Caixa Alta'] = df_textos['Texto'].apply(contar_palavras_caixa_alta)

# Exibindo os resultados
print("DataFrame de textos para teste:\n", df_textos)
print("\nDataFrame com os resultados das funções:\n", df_resultados)
