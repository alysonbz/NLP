import re

# Exemplo 1: o padrão "mineracao" aparece no início da string
resp = re.match('mineracao', 'mineracao de dados')
print(resp)
# Saída: <re.Match object; span=(0, 9), match='mineracao'>

# Exemplo 2: o padrão "dados" não aparece no início da string
resp = re.match('dados', 'mineracao de dados')
print(resp)
# Saída: None


print("---------------------------------------------------")

# Exemplo 1: correspondência no início da string
word_regex = r"\w+"
resp = re.match(word_regex, "semana de aula")
print(resp)
# Saída: <re.Match object; span=(0, 6), match='semana'>

# Exemplo 2: a string começa com um espaço, então o primeiro caractere que corresponde é 's'
word_regex = r"\w+"
resp = re.match(word_regex, "s emana de aula")
print(resp)
# Saída: <re.Match object; span=(0, 1), match='s'>


print("---------------------------------------------------")

# Exemplo 1: \w corresponde a um caractere de palavra (letra, número ou underscore)
word_regex = r"\w"
resp = re.match(word_regex, "semana de aula")
print(resp)
# Saída: <re.Match object; span=(0, 1), match='s'>

# Exemplo 2: \d corresponde a um dígito (0–9)
word_regex = r"\d"
resp = re.match(word_regex, "42semana de aula")
print(resp)
# Saída: <re.Match object; span=(0, 1), match='4'>

# Exemplo 3: \d+ corresponde a um ou mais dígitos consecutivos
word_regex = r"\d+"
resp = re.match(word_regex, "42semana de aula")
print(resp)
# Saída: <re.Match object; span=(0, 2), match='42'>

print("-----------------------------------------------------")

# Exemplo 1: busca por uma palavra no início da string
padrao = r'Universidade'
texto = "universidade federal do ceara"

resultado = re.match(padrao, texto)

if resultado:
    print("Padrão encontrado no início da string.")
else:
    print("Padrão não encontrado no início da string.")


# Exemplo 2: validação simples de formato de CPF
def validar_cpf(cpf):
    padrao = r'^\d{3}\.\d{3}\.\d{3}-\d{2}$'
    return bool(re.match(padrao, cpf))

cpf = "123.456.789-00"

if validar_cpf(cpf):
    print(f"{cpf} é um CPF válido.")
else:
    print(f"{cpf} não é um CPF válido.")

print("--------------------------------------------")

# Exemplo 1: dividir o texto por espaços em branco
word_regex = r"\s+"
resp = re.split(word_regex, "semana de aula")
print(resp)
# Saída: ['semana', 'de', 'aula']

# Exemplo 2: dividir o texto usando o ponto de exclamação "!"
word_regex = r"\!"
resp = re.split(word_regex, "semana quente! de aula")
print(resp)
# Saída: ['semana quente', ' de aula']


print("-----------------------------------------------")

# Exemplo 1: dividir o texto removendo letras minúsculas (a-z)
word_regex = r"[a-z]"
resp = re.split(word_regex, "Semana Quente! De Aula")
print(resp)
# Saída: ['S', '', '', '', ' ', 'Q', '', '', '', '! D', ' A', '', '', '', '']

# Exemplo 2: dividir o texto removendo letras minúsculas e também caracteres de palavra (\w)
word_regex = r"[a-z]|\w+"
resp = re.split(word_regex, "4 Semanas Quente! De Aula")
print(resp)
# Saída: ['4 S', ' Q', '! De A', '']

print("------------------------------------------------")

# Exemplo de uso do re.findall()
word_regex = r"[a-z]\w+"
resp = re.findall(word_regex, "4 Semanas Quente! De Aula")
print(resp)
# Saída: ['emanas', 'uente', 'ula']

print("-------------------------------------------------")

padrao = r'\d+'
texto = "123 abc 456 def"

numeros = re.findall(padrao, texto)

print("Números encontrados:", numeros)
# Saída: Números encontrados: ['123', '456']

print("------------------------------------------------")

padrao = r"ceara"
texto = "Universidade Federal do Ceara"

resultado = re.search(padrao, texto, flags=re.IGNORECASE)

if resultado:
    print("Padrão encontrado:", resultado.group())
else:
    print("Padrão não encontrado.")