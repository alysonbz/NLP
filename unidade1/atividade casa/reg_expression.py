#atividade data  04/04

#10 funções para análise de expressões regulares


import re

text = 'A linguagem Python é igual a Python que é´ Python + Python subtraido por Python Python'

# 1. Contagem de Correspondências
def contar_python(texto):
    return len(re.findall(r'\bPython\b', texto))

print(f'1. Contagem de "Python": {contar_python(text)}')



# 2. Validação de E-mail
em = 'alekyni@hotmail.com'
def validar_email(email):
    padrao = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(padrao, email))
print(f'2. Validação de E-mail: {validar_email(em)}')


# 3. Extração de Números de Telefone

tel = '85-996584932'
def extrair_telefones(texto):
    padrao = r'\d{2}[-]?\d{9}'
    return re.findall(padrao, texto)

print(f'3. Extração de Telefones: {extrair_telefones(tel)}')

# 4. Substituição de Palavras

tex = 'o gato'
def substituir_gato(texto):
    return re.sub(r'\bgato\b', 'cachorro', texto)

print(f'4. Substituição de "gato": {substituir_gato(tex)}')



# 5. Extração de URLs
tex = 'O link é https://www.google.com'

def extrair_urls(texto):
    padrao = r'https://(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'
    return re.findall(padrao, texto)

print(f'5. Extração de URLs: {extrair_urls(tex)}')

# 6. Verificação de Senha Segura
senha = 'AleK$8e0'

def verificar_senha(senha):
    padrao = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%?&])[A-Za-z\d@$!%?&]{8,}$'
    return bool(re.match(padrao, senha))

print(f'6. Verificação de Senha: {verificar_senha(senha)}')

# 7 extração de palavras
t = 'Mamute bebeu agua'
def extrair_palavras(texto):
  return re.findall(r'\b\w+\b', texto)
print(f'7. Extração de Palavras: {extrair_palavras(t)}')

# 8 Validação de datas
date = '05/04/2023'
def valid_data(texto):
  padrao = r'\d{2}/\d{2}/\d{4}'
  return re.findall(padrao, texto)
print(f'8. Validação de Datas: {valid_data(date)}')


# 9. Extração de Nomes Próprios
t = 'Aleksyni é um nome próprio'
def extrair_nomes_proprios(texto):
    return re.findall(r'\b[A-Z][a-z]*\b', texto)
print(f'9. Extração de Nomes Próprios: {extrair_nomes_proprios(t)}')



# 10 cont_vogais

vogais = 'a, y, i, o, b, c, v'
def contagem_vogais(texto):
  padrao = r'[aeiouAEIOU]'
  return len(re.findall(padrao, texto))
print(f'10. Contagem de Vogais: {contagem_vogais(vogais)}')
