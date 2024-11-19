import pandas as pd
import re

# Atividade
# 1) Um função que retorne a quantidade de sentenças em um texto.

def contar_sentencas(texto):
  padrao = r'[.!?]'
  sentencas = re.split(padrao, texto)
  sentencas = [sentenca for sentenca in sentencas if sentenca]
  return len(sentencas)

# Exemplo de uso:
texto = "Olá, como vai? Tudo bem. Até logo! Acabei de cair."
numero_sentencas = contar_sentencas(texto)



# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.

def contar_palavras_maiusc(texto):
  contador = 0
  for i in texto.split():
    # Verifica se a primeira letra é maiúscula
    if i[0].isupper():
      contador += 1
  return contador

# Exemplo de uso:
texto = "Eu Sou Homen grande."
resultado = contar_palavras_maiusc(texto)
print(resultado)

# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.

def contar_char_numerico(texto):

    # Expressão regular para encontrar todos os dígitos
    padrao = r'\d'

    # Encontrar todas as ocorrências de dígitos
    digitos = re.findall(padrao, texto)

    # Retornar o número de dígitos encontrados
    return len(digitos)

# Exemplo de uso:
texto = 'comi 9 pães, e 4 2 3  sanduiches'
resultado = contar_char_numerico(texto)
print(resultado)

# 4) Uma função que retorne a quantidade de palavras que estão  em caixa alta.

def contar_palavras_caixa_alta(texto):
  palavras = texto.split()
  contador = 0

  for palavra in palavras:
    if palavra.isupper():
      contador += 1

  return contador

texto = "Eu AMO Python. MAS NÃO GOSTO DE JAVA"
results = contar_palavras_caixa_alta(texto)


# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.

textos = [
    "Olá, como vai? Tudo bem. Até logo! Acabei de cair.",
    "Eu Sou Homen grande.",
    'comi 9 pães, e 4 2 3  sanduiches',
    "Eu AMO Python. MAS NÃO GOSTO DE JAVA"
]

# Criando um DataFrame com uma coluna 'texto'
df = pd.DataFrame({'texto': textos})

print(df)
# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.

# Criando o DataFrame e aplicando as funções
df = pd.DataFrame({'texto': textos})
df['num_sentencas'] = df['texto'].apply(contar_sentencas)
df['num_palavras_maiusc'] = df['texto'].apply(contar_palavras_maiusc)
df['num_caracteres_numericos'] = df['texto'].apply(contar_char_numerico)
df['num_palavras_caixa_alta'] = df['texto'].apply(contar_palavras_caixa_alta)

print(df.head())