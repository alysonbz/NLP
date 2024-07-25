# Atividade
# Um função que retorne a quantidade de sentenças em um texto.
import re

def contar_sentencas(texto):
    # Utiliza expressão regular para dividir o texto em sentenças
    sentencas = re.split(r'[.!?]+', texto)
    # Remove strings vazias da lista (ocorre quando há pontos consecutivos)
    sentencas = [sentenca for sentenca in sentencas if sentenca.strip()]
    # Retorna o número de sentenças encontradas
    return len(sentencas)

# Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
def contar_palavras_maiusculas(texto):
    # Divide o texto em palavras
    palavras = texto.split()
    # Inicializa o contador de palavras maiúsculas
    contador = 0
    # Itera sobre cada palavra e verifica se começa com letra maiúscula
    for palavra in palavras:
        if palavra[0].isupper():  # Verifica se a primeira letra é maiúscula
            contador += 1
    return contador

# Uma função que retorne a quantidade de caracteres numéricos em um texto.
def contar_caracteres_numericos(texto):
    # Inicializa o contador de caracteres numéricos
    contador = 0
    # Itera sobre cada caractere no texto
    for caractere in texto:
        if caractere.isdigit():  # Verifica se o caractere é numérico
            contador += 1
    return contador

# Uma função que retorne a quantidade de palavras que estão  em caixa alta.
def contar_palavras_caixa_alta(texto):
    # Divide o texto em palavras
    palavras = texto.split()
    # Inicializa o contador de palavras em caixa alta
    contador = 0
    # Itera sobre cada palavra e verifica se está em caixa alta
    for palavra in palavras:
        if palavra.isupper():  # Verifica se toda a palavra está em caixa alta
            contador += 1
    return contador



#######Exemplo de uso das funções:########
texto_exemplo = """
Este é um exemplo de texto. Contém várias sentenças. Algumas palavras estão em caixa alta.
Outras palavras têm a primeira letra maiúscula. Existem 123 números espalhados pelo texto.
"""

print("Número de sentenças:", contar_sentencas(texto_exemplo))
print("Palavras que começam com letra maiúscula:", contar_palavras_maiusculas(texto_exemplo))
print("Caracteres numéricos:", contar_caracteres_numericos(texto_exemplo))
print("Palavras em caixa alta:", contar_palavras_caixa_alta(texto_exemplo))


