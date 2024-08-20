### 1. Contagem de Correspondências: Escreva um programa que conte quantas vezes a palavra "Jão" aparece em uma determinada string usando expressões regulares.
"""

import re

texto = ["A vida com Jão é intensa, cheia de emoções profundas e letras que tocam o coração. Ele é um cantor que consegue transformar sentimentos em músicas, trazendo à tona tudo aquilo que às vezes não conseguimos expressar."]
padrao = r"Jão"

achados = re.compile(padrao)
cont = 0

for i in texto:
    achado = achados.findall(i)
    cont += len(achado)

print("Jão aparece", cont, "vezes")

"""### 2. Validação de E-mail: Crie uma função que valide se um dado texto representa um endereço de e-mail válido usando expressões regulares."""

def validar_email(email):
    padrao = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(padrao, email) is not None

emails_teste = ["teste@gmail.com", "invalido@gmail,com"]

for email in emails_teste:
    print(f"{email}: {validar_email(email)}")

"""### 3. Extração de Números de Telefone: Escreva um programa que extraia todos os números de telefone de um texto usando expressões regulares."""

def extrair_numeros_telefone(texto):
    padrao = r'\(\d{2}\)\s\d{4}-?\d{5}'
    return re.findall(padrao, texto)

texto_teste = "(86) 991327577"

numeros_telefone = extrair_numeros_telefone(texto_teste)
print("Números encontrados:")
for numero in numeros_telefone:
    print(numero)

"""### 4. Substituição de Palavras: Crie uma função que substitua todas as ocorrências de "gata" por "cadela" em um texto usando expressões regulares."""

def substituir_palavras(texto):
    padrao = r'\bgata\b'
    texto_substituido = re.sub(padrao, 'cadela', texto)
    return texto_substituido

texto_teste = "A Vanessa tem uma moto vermelha."

texto_substituido = substituir_palavras(texto_teste)
print("Texto original:")
print(texto_teste)
print("\nTexto substituído:")
print(texto_substituido)

"""### 5. Extração de URLs: Escreva um programa que extraia todas as URLs de um texto usando expressões regulares."""

def extrair_url(texto):
    padrao = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(padrao, texto)

site_teste = "http://www.algumacoisa.com/"

urls_extraidas = extrair_url(site_teste)
print(urls_extraidas)

"""### 6. Verificação de Senha Segura: Crie uma função que valide se uma senha é segura ou não usando expressões regulares. Considere uma senha segura se tiver pelo menos 8 caracteres, incluindo letras maiúsculas, minúsculas, números e caracteres especiais."""

def verificar_senha_segura(senha):
    if len(senha) < 8:
        return False
    if (re.search(r'[A-Z]', senha) and
        re.search(r'[a-z]', senha) and
        re.search(r'\d', senha) and
        re.search(r'[@$!%*?&#]', senha)):
        return True
    else:
        return False

senhas_teste = ["Senha123!", "senha123"]

for senha in senhas_teste:
    print(f"{senha}: {verificar_senha_segura(senha)}")

"""### 7. Extração de Palavras: Escreva uma função que extraia todas as palavras de uma string usando expressões regulares."""

def extrair_palavras(texto):
    padrao = r'\b\w+\b'
    return re.findall(padrao, texto)

texto_teste = "Escreva uma função que extraia todas as palavras de uma string usando expressões"

palavras_extraidas = extrair_palavras(texto_teste)
print("Palavras extraídas:")
print(palavras_extraidas)

"""### 8. Validação de Data: Crie uma função que valide se uma data está no formato "dd/mm/aaaa" usando expressões regulares."""

def validar_data(data):
    padrao = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/[0-9]{4}$'
    return re.match(padrao, data) is not None

datas_teste = ["23/01/2019", "02/09/2002"]

for data in datas_teste:
    print(f"{data}: {validar_data(data)}")

"""### 9. Extração de Nomes Próprios: Escreva um programa que extraia todos os nomes próprios (palavras iniciadas por maiúsculas) de um texto usando expressões regulares."""

def extrair_nomes_proprios(texto):
    padrao = r'\b[A-Z][a-zA-Z]*\b'
    return re.findall(padrao, texto)

texto_teste = "Davi, Thais e Vanessa são amigos de Bianca, Thays, Bruna, Tiago e Eryka"

nomes_proprios = extrair_nomes_proprios(texto_teste)
print("Nomes próprios extraídos:")
print(nomes_proprios)

"""### 10. Contagem de Vogais: Crie uma função que conte o número de vogais em uma string usando expressões regulares."""

def contar_vogais(texto):
    padrao = r'[aeiouAEIOU]'
    vogais_encontradas = re.findall(padrao, texto)
    return len(vogais_encontradas)

texto_teste = "Alguma coisa com vogais, palavras com vogais"

numero_vogais = contar_vogais(texto_teste)
print(f"Número de vogais: {numero_vogais}")
