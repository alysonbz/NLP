import pandas as pd
import re


# Atividade
texto_teste = "3 Anéis para os Reis-Elfos sob este céu, 7 para os Senhores-Anões em suas salas de pedra, 9 para os Homens Mortais fadados ao túmulo, 1 para o SENHOR DO ESCURO em seu trono sombrio."

# 1) Um função que retorne a quantidade de sentenças em um texto.
def contar_sentencas(texto):
    sentencas = re.split(r'[.!?]\s*', texto) # ponto final, exclamação ou interrogação seguidos por espaço ou fim de texto
    sentencas = [s for s in sentencas if s]
    return len(sentencas)

print('Questão 1 ', '-'*90, '\n',
      contar_sentencas(texto_teste))

# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
def contar_palavras_maiusculas(texto):
    palavras = texto.split()
    count = sum(1 for palavra in palavras if palavra[0].isupper())
    return count

print('Questão 2 ', '-'*90, '\n',
      contar_palavras_maiusculas(texto_teste))

# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.
def contar_caracteres_numericos(texto):
    return sum(c.isdigit() for c in texto)

print('Questão 2 ', '-'*90, '\n',
      contar_caracteres_numericos(texto_teste))

# 4) Uma função que retorne a quantidade de palavras que estão em caixa alta.
def contar_palavras_caixa_alta(texto):
    palavras = texto.split()
    count = sum(1 for palavra in palavras if palavra.isupper())
    return count

print('Questão 2 ', '-'*90, '\n',
      contar_palavras_caixa_alta(texto_teste))

# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.

df_questao_5 = pd.DataFrame([texto_teste]*4, columns=['Texto'])

print(df_questao_5)

# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.

df_questao_6 = pd.DataFrame(columns=['Qtde. de Sentenças',
                                     'Qtde. de palavras iniciando com maiúscula',
                                     'Qtde. caracteres numéricos',
                                     'Qtde. de palavras em caixa alta'])

df_questao_6['Qtde. de Sentenças'] = df_questao_5['Texto'].apply(contar_sentencas)
df_questao_6['Qtde. de palavras iniciando com maiúscula'] = df_questao_5['Texto'].apply(contar_palavras_maiusculas)
df_questao_6['Qtde. caracteres numéricos'] = df_questao_5['Texto'].apply(contar_caracteres_numericos)
df_questao_6['Qtde. de palavras em caixa alta'] = df_questao_5['Texto'].apply(contar_palavras_caixa_alta)

print(df_questao_6.T)