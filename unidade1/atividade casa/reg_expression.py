import re

resp = re.match(´mineracao´, ´mineracao de dados´)
print(resp)

#

word_regex = "\W+"
resp = re.match(word_regex, "semana de aula")
print(resp)

#

word_regex = "\W+"
resp = re.match(word_regex, "s emana de aula")
print(resp)

#

word_regex = "\W"
resp = re.match(word_regex, "semana de aula")
print(resp)

#

word_regex = "\d"
resp = re.match(word_regex, "42semana de aula")
print(resp)

#

word_regex = "\d+"
resp = re.match(word_regex, "42semana de aula")
print(resp)

#

padrao = r'Universidade'
texto = "universidade federal do ceara"
resultado = re.match(padrao, texto)

if resultado:
    print("Padrão encontrado no início da string.")
else:
    print("Padrão não encontradp no início da string")

#

def validar_cpf(cpf):
    padrao = r'^\d{3}\.\d{3}\.\{3}-\d{3}-\d{2}$'
    return bool(re.match(padrao, cpf))

cpf = "123.456.789-00"
if validar_cpf(cpf):
    print(f"{cpf} é um CPF válido.")
else:
    print(f"{cpf} não é um CPF válido.")

#

word_regex = "\s+"
resp = re.split(word_regex, "semana de aula")
print(resp)

#

word_regex = r"\!"
resp = re.split(word_regex, "semana quente! de aula")
print(resp)

#

word_regex = r"[a-z]"
resp = re.split(word_regex, "Semana Quente! De Aula")
print(resp)

#

word_regex = r"[a-z]\W+"
resp = re.split(word_regex, "4 Semana Quente! De Aula")
print(resp)

#

word_regex = r"[a-z]\W+"
resp = re.findall(word_regex, "4 Semana Quente! De Aula")
print(resp)

#

padrao = r'\d+'
texto = "123 abc 456 def"

numeros = re.findall(padrao, texto0)

print("Números encontrados: ", numeros)