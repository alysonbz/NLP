import pandas as pd
import re

# Atividade
# 1) Um função que retorne a quantidade de sentenças em um texto.
def contar_sentencas(texto):
    # Considera ., ! e ? como final de sentença
    sentencas = re.split(r'[.!?]+', texto)
    # Remove sentenças vazias
    sentencas = [s.strip() for s in sentencas if s.strip()]
    return len(sentencas)

# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
def contar_palavras_maiuscula_inicial(texto):
    palavras = texto.split()
    return sum(1 for p in palavras if p[0].isupper())

# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.
def contar_numeros(texto):
    return sum(c.isdigit() for c in texto)

# 4) Uma função que retorne a quantidade de palavras que estão  em caixa alta.
def contar_palavras_caixa_alta(texto):
    palavras = texto.split()
    return sum(1 for p in palavras if p.isupper())

# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.
textos = [
    "Olá! Meu nome é Pedro. Eu gosto de programar em Python.",
    "O CÃO CORREU atrás do CARRO 2 vezes!",
    "Hoje é dia 28/10/2025. Está um belo dia!",
    "A NASA e o FBI estão em missão. O foguete APOLLO foi lançado!"
]
df_textos = pd.DataFrame(textos, columns=["texto"])

# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.
df_resultados = pd.DataFrame({
    "Qtd_Sentenças": df_textos["texto"].apply(contar_sentencas),
    "Qtd_Palavras_Maiúscula_Inicial": df_textos["texto"].apply(contar_palavras_maiuscula_inicial),
    "Qtd_Números": df_textos["texto"].apply(contar_numeros),
    "Qtd_Palavras_Caixa_Alta": df_textos["texto"].apply(contar_palavras_caixa_alta)
})

print("DataFrame com os textos originais:")
print(df_textos, "\n")

print("DataFrame com os resultados das extrações:")
print(df_resultados)