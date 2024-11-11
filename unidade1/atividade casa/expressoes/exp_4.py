import re

def substituir(texto):
    return  re.sub(r'\bgato\b', 'cachorro', texto)

texto = 'se não tem cachorro, caça com gato'
print(substituir(texto))