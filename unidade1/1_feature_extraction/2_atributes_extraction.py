import pandas as pd
# Atividade
# 1. Um função que retorne a quantidade de sentenças em um texto.
def conta_palavra(texto):
  palavra = texto.split()
  return len(palavra)

teste_texto = "Alguma coisa escrita aqui, algo mais"
print(conta_palavra(teste_texto))

# 2. Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
def quantidade_palavras_mais(texto):
  palavras = texto.split()
  contador = 0
  for palavra in palavras:
      if palavra[0].isupper():
        contador += 1
  return contador

teste_texto = "Alguma coisa escrita aqui. Algo mais"
print(quantidade_palavras_mais(teste_texto))

#3. Uma função que retorne a quantidade de caracteres numéricos em um texto.
def qnt_num(texto):
  num = 0
  for char in texto:
    if char.isnumeric():
      num += 1
  return num
teste_texto = "Algum4 co1sa escrita aqui. Alg0 mais"
print(qnt_num(teste_texto))

# 4. Uma função que retorne a quantidade de palavras que estão  em caixa alta.
def qnt_palavras_cx_alta(texto):
  palavras = texto.split()
  contador = 0
  for palavra in palavras:
    if palavra.isupper():
      contador += 1
  return contador
teste_texto = "Alguma COISA escrita aqui. Algo MAIS"
print(qnt_palavras_cx_alta(teste_texto))

# 5. Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.
lista_texto = ["Alguma coisa escrita aqui, algo mais",
               "Alguma coisa escrita aqui. Algo mais",
               "Algum4 co1sa escrita aqui. Alg0 mais",
               "Alguma COISA escrita aqui. Algo MAIS"]

df = pd.DataFrame(lista_texto, columns=['texto'])

# 6. Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.

df['Number of sentences'] = df['texto'].apply(conta_palavra)
df['Number of capitalized words'] = df['texto'].apply(quantidade_palavras_mais)
df['Number of numeric characters'] = df['texto'].apply(qnt_num)
df['Number of uppercase words'] = df['texto'].apply(qnt_palavras_cx_alta)
