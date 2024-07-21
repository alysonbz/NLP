# Bibliotecas
import nltk 
import re
import polars as pl

# Atividade

# Um função que retorne a quantidade de sentenças em um texto.


texto = """
Olá. Meu nome é Bruna. Tenho 21 anos.
Amanhã vou viajar dia 16. UHUL VAMOS LÁ
"""

def qtd_frases(texto):
    sentencas = nltk.sent_tokenize(texto)
    return len(sentencas)


# Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.

def qtd_letras_maisculas(texto):
    palavras = re.findall(r'\b[A-Z][a-zA-Z]*\b', texto)
    return len(palavras)


# Uma função que retorne a quantidade de caracteres numéricos em um texto.

def qtd_carac_num(texto):
    palavras = re.findall(r'\d+', texto)
    return len(palavras)


# Uma função que retorne a quantidade de palavras que estão  em caixa alta.

def qtd_caixa_alta(texto):
    palavras = texto.split()
    palavras = [palavra for palavra in palavras if palavra.isupper()]
    return len(palavras)

# Tabela
df = pl.DataFrame({'Texto' : texto})

df = df.with_columns([
    pl.col('Texto').map_elements(qtd_frases).alias('Número de frases'),
    pl.col('Texto').map_elements(qtd_letras_maisculas).alias('Quantidade de palavras que começam com letra maiúscula'),
    pl.col('Texto').map_elements(qtd_carac_num).alias('Quantidade de caracteres numéricos'),
    pl.col('Texto').map_elements(qtd_caixa_alta).alias('Quantidade de palavras em caixa alta')
])

print(df)