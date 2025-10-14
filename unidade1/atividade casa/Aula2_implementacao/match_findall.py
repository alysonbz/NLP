import re

padrao = r'\d+'
texto = '123 abc 456 def'

numeros = re.findall(padrao, texto)

print(f'Números encontrados: {numeros}')
