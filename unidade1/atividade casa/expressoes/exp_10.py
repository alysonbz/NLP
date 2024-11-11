
import re

def contar_vogais(texto):
    return len(re.findall(r'[aeiouAEIOU]', texto))

texto = "Python é interessante."
print(contar_vogais(texto))