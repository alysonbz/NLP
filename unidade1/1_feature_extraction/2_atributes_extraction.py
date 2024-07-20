# Atividade
# Um função que retorne a quantidade de sentenças em um texto.
# Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
# Uma função que retorne a quantidade de caracteres numéricos em um texto.
# Uma função que retorne a quantidade de palavras que estão  em caixa alta.
#questao1
import re


import re

def contar_sentencas(texto):
    sentencas = re.split(r'[.!?]', texto)
    # Remove sentenças vazias (geradas quando há dois pontos seguidos)
    sentencas = [sentenca.strip() for sentenca in sentencas if sentenca.strip()]
    return len(sentencas)

texto = 'brasil eh grande. copa do mundo'
quantidade_sentencas = contar_sentencas(texto)
print(quantidade_sentencas)
#questao2
def contar_palavras_maiusculas(texto):
    palavras = texto.split()
    contador = sum(1 for palavra in palavras if palavra and palavra[0].isupper())
    return contador

texto='Brasil te amo. Brasil te amamos'
maiusculas = contar_palavras_maiusculas(texto)
print(maiusculas)

def contar_caracteres_numericos(texto):
    contador = sum(1 for caractere in texto if caractere.isdigit())
    return contador
texto='Brasil te amo. Brasil te amamos. 1'
numeros = contar_caracteres_numericos(texto)
print(numeros)

def contar_palavras_caixa_alta(texto):
    palavras = texto.split()
    contador = sum(1 for palavra in palavras if palavra.isupper())
    return contador

# Exemplo de uso:
print(contar_palavras_caixa_alta("Esta é uma STRING de EXEMPLO com várias VOGAIS. Brasil braSil Brasil"))

import pandas as pd

# Textos para teste
textos = [
    'Brasil é um país incrível. Eu amo o Brasil!',
    'A tecnologia move o mundo.',
    'Python 3.9.2 é a última versão disponível.',
    'OPENAI é uma organização de pesquisa em IA.'
]

# Criar um DataFrame de 4 linhas e 5 colunas
df = pd.DataFrame(index=range(1, 5), columns=['Texto', 'Sentenças', 'Palavras Maiúsculas', 'Caracteres Numéricos',
                                              'Palavras em Caixa Alta'])

# Preencher o DataFrame com os textos e os resultados das funções
for i, texto in enumerate(textos, start=1):
    print(f"Processando texto {i}: '{texto}'")
    df.loc[i, 'Texto'] = texto
    sentencas = contar_sentencas(texto)
    maiusculas = contar_palavras_maiusculas(texto)
    numeros = contar_caracteres_numericos(texto)
    caixa_alta = contar_palavras_caixa_alta(texto)

    print(f"Sentenças encontradas: {sentencas}")
    print(f"Palavras com letra maiúscula: {maiusculas}")
    print(f"Caracteres numéricos encontrados: {numeros}")
    print(f"Palavras em caixa alta: {caixa_alta}")

    # Preenchendo o DataFrame com os resultados
    df.loc[i, 'Sentenças'] = sentencas
    df.loc[i, 'Palavras Maiúsculas'] = maiusculas
    df.loc[i, 'Caracteres Numéricos'] = numeros
    df.loc[i, 'Palavras em Caixa Alta'] = caixa_alta
    print("---")

# Exibir o DataFrame final
print("\nDataFrame Final:")
print(df)
