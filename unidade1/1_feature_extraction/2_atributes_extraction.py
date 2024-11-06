import re
import pandas as pd
# Atividade

# 1) Um função que retorne a quantidade de sentenças em um texto.
def quanti_sentencas(texto1):
    regex = r'[^.!?]+[.!?]'
    resp = re.findall(regex,texto1)
    return len(resp)
texto1 = "olá, meu nome é Júlia. Sou natural de Itapajé."
print(quanti_sentencas(texto1))

# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
def quanti_maiusculas(texto2):
    resp = re.findall(r'[A-Z][a-z]*\b', texto2)
    return len(resp)
texto2 = "O Rato roeu a roupa do Rei de Roma"
print(quanti_maiusculas(texto2))

# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.
def quanti_caracteres(texto3):
    resp = re.findall(r'\d', texto3)
    return len(resp)
texto3 = "Eu tenho 2 gatos, 3 cachorros e 10 pássaros"
print(quanti_caracteres(texto3))

# 4) Uma função que retorne a quantidade de palavras que estão  em caixa alta.
def quanti_caixalta(texto4):
    resp = re.findall(r'[A-Z]+\b', texto4)
    return len(resp)
texto4 = "NÃO MEXA! pode quebrar, é FRÁGIL"
print(quanti_caixalta(texto4))

# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.
df = pd.DataFrame({"texto": [texto1, texto2, texto3, texto4]})
print(df)

# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.
df_resultados = pd.DataFrame({
    "Sentenças": df["texto"].apply(quanti_sentencas),
    "Palavras com maiúscula": df["texto"].apply(quanti_maiusculas),
    "Números": df["texto"].apply(quanti_caracteres),
    "Palavras em caixa alta": df["texto"].apply(quanti_caixalta)
})

print(df_resultados)
