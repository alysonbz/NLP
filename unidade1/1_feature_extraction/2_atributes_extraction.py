import pandas as pd

# Função para contar sentenças em um texto
def contar_sentencas(texto):
    sentencas = re.split(r'[.!?]', texto)
    return len(sentencas)

# Função para contar palavras que começam com letra maiúscula em um texto
def contar_palavras_maiusculas(texto):
    palavras = texto.split()
    contador = 0
    for palavra in palavras:
        if palavra and palavra[0].isupper():
            contador += 1
    return contador

# Função para contar caracteres numéricos em um texto
def contar_caracteres_numericos(texto):
    contador = 0
    for caractere in texto:
        if caractere.isdigit():
            contador += 1
    return contador

# Função para contar palavras em caixa alta em um texto
def contar_palavras_caixa_alta(texto):
    palavras = texto.split()
    contador = 0
    for palavra in palavras:
        if palavra.isupper():
            contador += 1
    return contador

# Textos de exemplo
textos = [
    "Oi, tudo bem?, como voce esta hj? espero",
    "Este é um Exemplo de Texto. Aqui Temos Algumas Palavras com Letra Maiúscula.",
    "Oi tudo, vc esta na cadeira 10.",
    "Este TEXTO possui ALGUMAS palavras EM caixa ALTA."
]

# Criando o DataFrame com uma coluna chamada 'Texto'
df = pd.DataFrame({'Texto': textos})

# Aplicando as funções a cada célula do DataFrame
df['Sentencas'] = df['Texto'].apply(contar_sentencas)
df['Palavras_Maiusculas'] = df['Texto'].apply(contar_palavras_maiusculas)
df['Caracteres_Numericos'] = df['Texto'].apply(contar_caracteres_numericos)
df['Palavras_Caixa_Alta'] = df['Texto'].apply(contar_palavras_caixa_alta)

# Exibindo o DataFrame
print(df)
