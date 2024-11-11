import re

def extrair_nomes_proprios(texto):
    return re.findall(r'\b[A-Z][a-z]*\b', texto)

texto = "Maria e João foram ao parque com Alice."
print(extrair_nomes_proprios(texto))