import pandas as pd

# 1. Função que retorne a quantidade de sentenças em um texto
def conta_sentencas(texto):
    sentencas = texto.split('. ')
    sentencas += texto.split('! ')
    sentencas += texto.split('? ')
    return len([s for s in sentencas if s.strip() != ''])

# 2. Função que retorne a quantidade de palavras que começam com letra maiúscula em um texto
def quantidade_palavras_mais(texto):
    palavras = texto.split()
    contador = 0
    for palavra in palavras:
        if palavra[0].isupper():
            contador += 1
    return contador

# 3. Função que retorne a quantidade de caracteres numéricos em um texto
def qnt_num(texto):
    num = 0
    for char in texto:
        if char.isnumeric():
            num += 1
    return num

# 4. Função que retorne a quantidade de palavras que estão em caixa alta
def qnt_palavras_cx_alta(texto):
    palavras = texto.split()
    contador = 0
    for palavra in palavras:
        if palavra.isupper():
            contador += 1
    return contador

# 5. Criar um DataFrame com os textos testados
lista_texto = [
    "A MINHA irmã Ester arrancou 2 dentinhos." ,
    "Eu faço aniversário dia 02 de setembro e completo 22 anos.",
    "Vou comprar um livro dia 30!",
    "HAHA LEVEI UM PEQUENO ESCORREGÃO!"
]

df = pd.DataFrame(lista_texto, columns=['texto'])

# 6. Aplicar as funções e criar novas colunas no DataFrame
df['Number of sentences'] = df['texto'].apply(conta_sentencas)
df['Number of capitalized words'] = df['texto'].apply(quantidade_palavras_mais)
df['Number of numeric characters'] = df['texto'].apply(qnt_num)
df['Number of uppercase words'] = df['texto'].apply(qnt_palavras_cx_alta)

print(df)
