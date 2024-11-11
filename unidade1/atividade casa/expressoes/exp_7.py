import re

def extrair(texto):
    return  re.findall(r'\bw+\b', texto)

texto = 'Python é uma linguagem de programação.'
print(extrair(texto))