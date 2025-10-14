import re

def validar_cpf(cpf):
    padrao = r'^\d{3}\.\d{3}\.\d{3}\-\d{2}$'
    return bool(re.match(padrao, cpf))

cpf = '123.456.789-00'
if validar_cpf(cpf):
    print(f'{cpf} é válido')
else:
    print(f'{cpf} não é válido')