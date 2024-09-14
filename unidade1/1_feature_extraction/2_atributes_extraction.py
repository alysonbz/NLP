import pandas as pd
import re

# 1) Função que retorna a quantidade de sentenças em um texto
def contar_sentencas(texto):
    sentencas = re.split(r'[.!?]+', texto)
    return len([s for s in sentencas if s.strip() != ''])

# 2) Função que retorna a quantidade de palavras que começam com letra maiúscula em um texto
def contar_palavras_maiusculas_inicio(texto):
    palavras = texto.split()
    return len([palavra for palavra in palavras if palavra[0].isupper()])

# 3) Função que retorna a quantidade de caracteres numéricos em um texto
def contar_caracteres_numericos(texto):
    return len([caractere for caractere in texto if caractere.isdigit()])

# 4) Função que retorna a quantidade de palavras que estão em caixa alta
def contar_palavras_caixa_alta(texto):
    palavras = texto.split()
    return len([palavra for palavra in palavras if palavra.isupper()])

# 5) Criação de um dataframe com 1 coluna e 4 linhas contendo textos para testar as funções
texto = [
    "A maternidade traz felicidade!",
    "A MATERNIDADE TRAZ FELICIDADE!",
    "Python Community 2024.1.6",
    "A data prevista do parto é 05 de setembro."
]

df_textos = pd.DataFrame({'Texto': texto})

# 6) Criação de um dataframe de 4 linhas e 4 colunas preenchido com os resultados das funções
df_resultados = pd.DataFrame({
    'Sentenças': df_textos['Texto'].apply(contar_sentencas),
    'Palavras com Maiúscula no Início': df_textos['Texto'].apply(contar_palavras_maiusculas_inicio),
    'Caracteres Numéricos': df_textos['Texto'].apply(contar_caracteres_numericos),
    'Palavras em Caixa Alta': df_textos['Texto'].apply(contar_palavras_caixa_alta)
})

# Print do dataframe criado
print(df_resultados)
