import re

# PRIMEIRA QUESTÃO

def contar_palavra(texto):
    padrao = r"python"    
    matches = re.findall(padrao, texto.lower())     
    quantidade = len(matches)
    return quantidade

texto = "Eu amo programar em Python. Python é a melhor linguagem. Já ouvi falar em Jython e CPython, mas Python é o meu foco."
print(contar_palavra(texto))



# SEGUNDA QUESTÃO
def validar_email(email):
    padrao = r'^\w+\@\w+'
    return bool(re.match(padrao,email))

email = "oaozinho123@gmail.com"

if validar_email(email):
    print("Email valido")
else:
    print("email invalido")


# TERCEIRA QUESTÃO
def extrair_numero(texto):
    padrao = r'\d{9}'
    resp = re.findall(padrao, texto)
    return resp

texto = "8899999999 foi amor a 5593999888 primeira 11 vista"
print(extrair_numero(texto))

# QUARTA QUESTÃO
def substituir(texto):
    padrao = r'gato'
    subs = r"cachorro"

    resp = re.sub(padrao, subs, texto)

    return resp

texto = "Eu acho que vi um gato em cima do muro"

print(substituir(texto))

# QUINTA QUESTÃO

def extrair_url(texto):
    padrao = r'https?://(?:[-\w] | (?:%[da-fA-F]{2}))+|www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

    url = re.findall(padrao, texto)
    return url

texto = 'https://www.joaozinho.com.br, testando url'
print(extrair_url(texto))

# SEXTA QUESTÃO

def validar_senha(texto):
    padrao_seguranca = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%^&*()_+=\-{}[\]:;<>,.?/|])(.{8,})$"
    
    if re.match(padrao_seguranca, texto):
        return True
    else:
        return False

senha1 = "Senha123@"
print(validar_senha(senha1))

# SETIMA QUESTÃO
def extrair_palavra(texto):
    padrao = r'\w+'

    palavra_encontrada = re.findall(padrao, texto)
    return palavra_encontrada

texto = "Teste de extração de palavras"
print(extrair_palavra(texto))


# OITAVA QUESTÃO
def validar_data(texto):
    padrao_data = r"^\d{2}/\d{2}/\d{4}$"

    if re.match(padrao_data, texto):
        return True
    else:
        return False
    
data = "27/04/2002"
print(validar_data(data))

# NONA QUESTÃO
def extrair_nome_proprio(texto):
    padrao_nome = r"[A-ZÀ-Ÿ][a-zà-ÿ]+\b"

    nome_encontrado = re.findall(padrao_nome, texto)

    return nome_encontrado

texto = "Joaozinho testou um novo Celular"
print(extrair_nome_proprio(texto))

# Decima questão
def contar_vogais(texto):
    padrao_vogais = r"[aeiouáéíóúàèìòùãõâêîôûAEIOUÁÉÍÓÚÀÈÌÒÙÃÕÂÊÎÔÛ]"
    vogais = re.findall(padrao_vogais, texto)
    return len(vogais)

texto = "Era uma bela tarde lá fora"
print(contar_vogais(texto))
    