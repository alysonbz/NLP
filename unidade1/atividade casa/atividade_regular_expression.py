# 1. Contagem de Correspondências
## Crie um programa que conte quantas vezes a palavra "Python" aparece em uma string.
import re

texto = "Estou aprendendo Python e Python é muito interessante."
contagem = len(re.findall(r'Python', texto))
print("Número de ocorrências de 'Python':", contagem)

print("___________________________________________________________")

# 2. Validação de E-mail
## Crie uma função que valide se um dado texto representa um e-mail válido.
def validar_email(email):
    padrao = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(padrao, email) is not None

email = "exemplo@dominio.com"
print(validar_email(email))

print("___________________________________________________________")

# 3. Extração de Números de Telefone
## Crie um programa que extraia todos os números de telefone de um texto.
texto = "Me liga no número (8) 98765-4321 ou (88) 99876-5432."
telefones = re.findall(r'\(\d{2}\)\s?\d{5}-\d{4}', texto)
print("Números encontrados:", telefones)

print("___________________________________________________________")

# 4. Substituição de Palavras
## Crie uma função que substitua todas as ocorrências de "gato" por "cachorro".
texto = "O gato está no telhado. O gato é branco."
texto_substituido = re.sub(r'gato', 'cachorro', texto)
print(texto_substituido)

print("___________________________________________________________")

# 5. Extração de URLs
## Crie um programa que extraia todas as URLs de um texto.
texto = "Visitei https://www.google.com e https://www.python.org para obter mais informações."
urls = re.findall(r'https?://[^\s]+', texto)
print("URLs encontradas:", urls)

print("___________________________________________________________")

# 6. Verificação de Senha Segura
## Crie uma função que valide se uma senha é segura ou não.
def verificar_senha(senha):
    padrao = r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[A-Z])(?=.*[a-z])(?=.*[!@#$%^&*]).{8,}$'
    return re.match(padrao, senha) is not None

senha = "SenhaForte@123"
print(verificar_senha(senha))

print("___________________________________________________________")

# 7. Extração de Palavras
## Crie uma função que extraia as palavras de uma string.
texto = "Eu gosto de Python, é muito legal."
palavras = re.findall(r'\b\w+\b', texto)
print("Palavras encontradas:", palavras)

print("___________________________________________________________")

# 8. Validação de Data
## Crie uma função que valide se uma data está no formato "dd/mm/aaaa".
def validar_data(data):
    padrao = r'^\d{2}/\d{2}/\d{4}$'
    return re.match(padrao, data) is not None

data = "15/10/2025"
print(validar_data(data))

print("___________________________________________________________")

# 9. Extração de Nomes Próprios
## Crie um programa que extraia todos os nomes próprios (palavras iniciadas por maiúsculas) de um texto.
texto = "Maria foi ao mercado."
nomes = re.findall(r'\b[A-Z][a-z]*\b', texto)
print("Nomes encontrados:", nomes)

print("___________________________________________________________")

# 10. Contagem de Vogais
## Crie uma função que conte o número de vogais em uma string.
def contar_vogais(texto):
    return len(re.findall(r'[aeiouáéêíóúAEIOUÁÉÍÓÚ]', texto))

texto = "Olá, como você está?"
print("Número de vogais:", contar_vogais(texto))

print("___________________________________________________________")

print("Atividade realizada dia 15/10 para apresentar dia 20 ")