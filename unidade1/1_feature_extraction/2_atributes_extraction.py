import pandas as pd
import re

# Função para contar sentenças em um texto
def contar_sentencas(texto):
    sentencas = re.split(r'[.!?]+', texto)
    # Remove elementos vazios resultantes do split
    sentencas = list(filter(lambda x: len(x.strip()) > 0, sentencas))
    return len(sentencas)

# Função para contar palavras que começam com letra maiúscula
def contar_palavras_maiusculas(texto):
    palavras = texto.split()
    contagem = sum(1 for palavra in palavras if palavra[0].isupper())
    return contagem

# Função para contar caracteres numéricos em um texto
def contar_caracteres_numericos(texto):
    return sum(1 for c in texto if c.isdigit())

# Função para contar palavras em caixa alta
def contar_palavras_caixa_alta(texto):
    palavras = texto.split()
    contagem = sum(1 for palavra in palavras if palavra.isupper())
    return contagem

# Textos de exemplo
textos = [
    "Esta é uma frase. Esta é outra frase!",
    "Python é uma linguagem de programação poderosa.",
    "12345 é um número grande.",
    "TEXTO EM CAIXA ALTA."
]

# 1) Criar um dataframe com os textos
df_textos = pd.DataFrame(textos, columns=['Texto'])

# 2) Criar um dataframe com os resultados das funções
df_resultados = pd.DataFrame()

# Aplicar as funções nos textos usando apply e armazenar os resultados no dataframe df_resultados
df_resultados['Sentencas'] = df_textos['Texto'].apply(contar_sentencas)
df_resultados['Palavras_Maiusculas'] = df_textos['Texto'].apply(contar_palavras_maiusculas)
df_resultados['Caracteres_Numericos'] = df_textos['Texto'].apply(contar_caracteres_numericos)
df_resultados['Palavras_Caixa_Alta'] = df_textos['Texto'].apply(contar_palavras_caixa_alta)

# Exibir o dataframe com os resultados
print(df_resultados)
