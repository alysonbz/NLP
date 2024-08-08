#atividade data  04/04

#10 funções para análise de expressões regulares

#Questão01
import re

def contar_ocorrencias_python(texto):
    padrao = r'\bPython\b'
    contagem = len(re.findall(padrao, texto))
    return contagem

# Exemplo de uso
texto = "Nós usamos Python para realização de projetos, a linguagem Python é uma das mais utilizadas atualmente "
ocorrencias = contar_ocorrencias_python(texto)
print("Número de ocorrências da palavra 'Python':", ocorrencias)

#Questão02

def validar_email(email):
    # Padrão de expressão regular para validar endereço de e-mail
    padrao = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    # Verifica se o email corresponde ao padrão
    if re.match(padrao, email):
        return True
    else:
        return False

# Exemplo de uso
endereco_email = "pedro@gmail.com"
if validar_email(endereco_email):
    print("O endereço de e-mail é válido.")
else:
    print("O endereço de e-mail é inválido.")

#Questão03


def extrair_numeros_telefone(texto):
    # Padrão de expressão regular para encontrar números de telefone
    padrao = r'\b\d{2,3}[-.\s]?\d{3,5}[-.\s]?\d{3,5}\b'
    # Usando findall para encontrar todas as correspondências
    numeros_telefone = re.findall(padrao, texto)
    return numeros_telefone

# Exemplo de uso
texto = """
Meu número de telefone é 123-456-7890.
Você pode me contatar no (123) 456-7890 ou no 123 456 7890.
Outro número de telefone é 987.654.3210.
"""
numeros = extrair_numeros_telefone(texto)
print("Números de telefone encontrados:")
for numero in numeros:
    print(numero)

#Questão04


def substituir_palavra(texto, palavra_antiga, palavra_nova):
    # Utiliza a função sub() do módulo re para substituir as ocorrências da palavra antiga pela nova
    novo_texto = re.sub(r'\b' + re.escape(palavra_antiga) + r'\b', palavra_nova, texto)
    return novo_texto

# Exemplo de uso
texto = "O gato pulou sobre o muro. O gato está na árvore."
texto_modificado = substituir_palavra(texto, "gato", "cachorro")
print("Texto original:")
print(texto)
print("\nTexto modificado:")
print(texto_modificado)

#Questão05


def extrair_urls(texto):
    # Padrão de expressão regular para encontrar URLs
    padrao = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    # Usando findall para encontrar todas as correspondências
    urls = re.findall(padrao, texto)
    return urls

# Exemplo de uso
texto = """
Aqui está um link para o meu site: https://www.exemplo.com.
Você também pode visitar meu blog em http://blog.exemplo.com.
Além disso, confira este link: https://www.outroexemplo.com/pagina.html.
"""
urls_encontradas = extrair_urls(texto)
print("URLs encontradas:")
for url in urls_encontradas:
    print(url)

#Questão06

def validar_senha(senha):
    # Padrão de expressão regular para validar a senha
    padrao = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    # Verifica se a senha corresponde ao padrão
    if re.match(padrao, senha):
        return True
    else:
        return False

# Exemplo de uso
senha1 = "SenhaSegura@123"
senha2 = "fraca123"
senha3 = "MaisFraca"
print("A senha1 é segura:", validar_senha(senha1))
print("A senha2 é segura:", validar_senha(senha2))
print("A senha3 é segura:", validar_senha(senha3))

#Questão07

def extrair_palavras(texto):
    # Padrão de expressão regular para encontrar palavras
    padrao = r'\b\w+\b'
    # Usando findall para encontrar todas as correspondências
    palavras = re.findall(padrao, texto)
    return palavras

# Exemplo de uso
texto = "O ceará é o melhor time do estado do Ceará!"
palavras_encontradas = extrair_palavras(texto)
print("Palavras encontradas:")
print(palavras_encontradas)

#Questão08

def validar_data(data):
    # Padrão de expressão regular para validar data no formato dd/mm/aaaa
    padrao = r'^(0[1-9]|[1-2][0-9]|3[0-1])/(0[1-9]|1[0-2])/\d{4}$'
    # Verifica se a data corresponde ao padrão
    if re.match(padrao, data):
        return True
    else:
        return False

# Exemplo de uso
data1 = "31/12/2022"
data2 = "30/02/2023"
data3 = "12/05/22"
print("A data1 é válida:", validar_data(data1))
print("A data2 é válida:", validar_data(data2))
print("A data3 é válida:", validar_data(data3))

#Questão09

def extrair_nomes_proprios(texto):
    # Padrão de expressão regular para encontrar nomes próprios
    padrao = r'\b[A-Z][a-z]+\b'
    # Usando findall para encontrar todas as correspondências
    nomes_proprios = re.findall(padrao, texto)
    return nomes_proprios

# Exemplo de uso
texto = "João encontrou Maria na Praça da República. Eles decidiram ir ao Cinema Rio."
nomes = extrair_nomes_proprios(texto)
print("Nomes próprios encontrados:")
print(nomes)

#Questão10
import re

def contar_vogais(texto):
    # Padrão de expressão regular para encontrar vogais
    padrao = r'[aeiouAEIOU]'
    # Usando findall para encontrar todas as correspondências
    vogais = re.findall(padrao, texto)
    # Retorna o número de vogais encontradas
    return len(vogais)

# Exemplo de uso
texto = "Olá mundo, Hello World"
quantidade_vogais = contar_vogais(texto)
print("Número de vogais:", quantidade_vogais)


