import re 

#match

resp = re.match("mineracao", "mineracao de dados")
print(resp)

resp1 = re.match("dados", "mineracao de dados")
print(resp1)

word_regex = "\w+"
resp2 = re.match(word_regex, "semana de aula")
print(resp2)

word_regex = "\w+"
resp3 = re.match(word_regex, "s emana de aula")
print(resp3)

word_regex = "\w"
resp4 = re.match(word_regex, "semana de aula")
print(resp4)

word_regex = "\d"
resp5= re.match(word_regex, "42semana de aula")
print(resp5)

word_regex = "\d+"
resp6 = re.match(word_regex, "42semana de aula")
print(resp6)

padrao = r'Universidade'
texto = "universidade federal do ceara"

resultado = re.match(padrao, texto)

if resultado:
    print("Padrão encontrado no início da string")
else:
    print("Padrão não encontrado no início da string")

def validar_cpf(cpf):
    padrao = r'^\d{3}\.\d{3}.\d{3}-.\d{2}$'
    return bool(re.match(padrao, cpf))

cpf = "123.456.789-00"
if validar_cpf(cpf):
    print(f"{cpf} é um cpf inválido")
else:
    print(f"{cpf} é um cpf válido")

#split

word_regex = "\s+"
resp7 = re.split(word_regex, "semana de aula")
print(resp7)

word_regex = r"\!"
resp8 = re.split(word_regex, "semana quente! de aula")
print(resp8)

word_regex = r"[a-z]"
resp9 = re.split(word_regex, "Semana Quente! De Aula")
print(resp9)

word_regex = r"[a-z]\w+"
resp0 = re.split(word_regex, "4 Semanas Quente! De Aula")
print(resp0)

#findall

word_regex = r"[a-z]\w+"
resp01 = re.findall(word_regex, "4 Semanas Quente! De Aula")
print(resp01)

padrao = r"\d+"
texto = "123 abc 456 def"

numeros = re.findall(padrao, texto)

print("Números encontrados:", numeros)