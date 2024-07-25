#atividade data  04/04
#questão 1
import re
#
def contar_ocorrencias_python(texto):
    padrao = r'Python'  # Expressão regular para encontrar a palavra "Python"
    ocorrencias = re.findall(padrao, texto)
    return len(ocorrencias)

# Exemplo de uso:
texto = "Python é uma linguagem de programação poderosa. Python é amplamente usada em várias aplicações. e cada vez mais o Python esta sendo utilizado"
resultado = contar_ocorrencias_python(texto)
print("Número de ocorrências de 'Python':", resultado)

#questão 2
import re

def validar_email_UFC(email):
    padrao = r'^[\w\.-]+@ufc\.br$'  # Expressão regular para validar um endereço de email da UFC
    if re.match(padrao, email):
        return True
    else:
        return False

# Solicitar ao usuário que insira
email_usuario = input("Por favor, insira o seu email: ")

# Verificar o
if validar_email_UFC(email_usuario):
    print("O email pertence à Universidade Federal do Ceará (UFC).")
else:
    print("O email não pertence à Universidade Federal do Ceará (UFC).")

#questão 3
import re


def extrair_numeros_telefone(texto):
    # Define o padrão para números de telefone
    padrao = r'\b\d{2,3}\s?\d{4,5}-?\d{4}\b'  # Padrão para números de telefone com ou sem hífen

    # Encontra todos os números de telefone no texto
    numeros_telefone = re.findall(padrao, texto)
    return numeros_telefone


# Solicitar ao usuário que insira o texto
texto_usuario = input("Por favor, insira o texto contendo os números de telefone: ")

# Extrair números de telefone do texto inserido pelo usuário
numeros_encontrados = extrair_numeros_telefone(texto_usuario)

# Imprimir os números de telefone encontrados
print("Números de telefone encontrados:")
for numero in numeros_encontrados:
    print(numero)

#questao4
import re


def substituir_gato_por_cachorro(texto):
    # Define o padrão para encontrar a palavra "gato"
    padrao = r'\bgato\b'  # \b é usado para corresponder à palavra "gato" como uma palavra inteira

    # Substitui todas as ocorrências de "gato" por "cachorro" no texto
    texto_substituido = re.sub(padrao, 'cachorro', texto)
    return texto_substituido


# Exemplo de uso:
texto = "O gato está dormindo no telhado. O gato é um animal doméstico."
texto_substituido = substituir_gato_por_cachorro(texto)
print(texto_substituido)

#questao5
import re


def extrair_urls(texto):
    # Define o padrão para encontrar URLs
    padrao = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

    # Encontra todas as URLs no texto
    urls_encontradas = re.findall(padrao, texto)
    return urls_encontradas


# Solicitar ao usuário que insira o texto
texto_usuario = input("Por favor, insira o texto contendo as URLs: ")

# Extrair URLs do texto inserido pelo usuário
urls_encontradas = extrair_urls(texto_usuario)

# Imprimir as URLs encontradas
print("URLs encontradas:")
for url in urls_encontradas:
    print(url)

#questao6
import re


def validar_senha(senha):
    """
    Função para validar se uma senha é segura ou não.

    Condições para uma senha segura:
    - Pelo menos 8 caracteres de comprimento.
    - Pelo menos uma letra minúscula.
    - Pelo menos uma letra maiúscula.
    - Pelo menos um dígito numérico.
    - Pelo menos um dos seguintes caracteres especiais: @, $, !, %, *, ?, &.
    """
    # Define o padrão para validar uma senha segura
    padrao = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'

    # Verifica se a senha corresponde ao padrão
    if re.match(padrao, senha):
        return True
    else:
        return False


# Exibir as condições para uma senha segura
print("Para uma senha ser segura, ela deve atender aos seguintes critérios:")
print("- Pelo menos 8 caracteres de comprimento.")
print("- Pelo menos uma letra minúscula.")
print("- Pelo menos uma letra maiúscula.")
print("- Pelo menos um dígito numérico.")
print("- Pelo menos um dos seguintes caracteres especiais: @, $, !, %, *, ?, &.")

# Solicitar ao usuário que insira a senha
senha_usuario = input("Por favor, insira a senha a ser validada: ")

# Verificar se a senha inserida é segura
if validar_senha(senha_usuario):
    print("A senha inserida é segura.")
else:
    print("A senha inserida não é segura. Por favor, certifique-se de que ela atenda aos critérios mencionados acima.")

#questao7
import re


def extrair_palavras(texto):
    # Define o padrão para encontrar palavras
    padrao = r'\b\w+\b'  # \b é usado para encontrar o limite de uma palavra

    # Encontra todas as palavras no texto
    palavras_encontradas = re.findall(padrao, texto)
    return palavras_encontradas


# Exemplo de uso:
texto = "Esta é uma string de exemplo, com várias palavras diferentes."
palavras = extrair_palavras(texto)
print(palavras)

#questao8
import re


def validar_formato_data(data):
    # Define o padrão para validar o formato da data "DD/MM/AA"
    padrao = r'^\d{2}/\d{2}/\d{2}$'

    # Verifica se a data corresponde ao padrão
    if re.match(padrao, data):
        return True
    else:
        return False


# Solicitar ao usuário que insira a data
data_usuario = input("Por favor, insira a data no formato 'DD/MM/AA': ")

# Verificar se a data inserida está no formato correto
if validar_formato_data(data_usuario):
    print("A data inserida está no formato correto 'DD/MM/AA'.")
else:
    print("A data inserida não está no formato correto 'DD/MM/AA'.")

#questao9
import re


def extrair_nomes_proprios(texto):
    # Define o padrão para encontrar nomes próprios
    padrao = r'\b[A-Z][a-z]*\b'  # Procura por palavras iniciadas com uma letra maiúscula seguida de letras minúsculas

    # Encontra todos os nomes próprios no texto
    nomes_proprios = re.findall(padrao, texto)
    return nomes_proprios


# Exemplo de uso:
texto = "Vitoria, Mateus, Arthur, Pedro e Caique foram ao jogo hoje."
nomes = extrair_nomes_proprios(texto)
print("Nomes próprios encontrados:")
print(nomes)


#questao10
import re


def contar_vogais(texto):
    # Define o padrão para encontrar vogais
    padrao = r'[aeiouAEIOU]'  # Procura por qualquer uma das vogais (maiúsculas ou minúsculas)

    # Encontra todas as vogais no texto
    vogais_encontradas = re.findall(padrao, texto)

    # Retorna o número de vogais encontradas
    return len(vogais_encontradas)


# Exemplo de uso:
texto = "Esta é uma frase com várias vogais."
total_vogais = contar_vogais(texto)
print("Total de vogais:", total_vogais)