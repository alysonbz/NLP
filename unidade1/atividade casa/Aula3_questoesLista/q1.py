# 1. Contagem de correspondências:

import re

texto = "Python é um de alto nível, linguagem de programação de uso geral. Sua filosofia de design enfatiza legibilidade do código com o uso de recuo significativo. Python é verificado dinamicamente por tipo e coletado como lixo. Ele suporta múltiplos paradigmas de programação, incluindo estruturado (particularmente processual), orientado a objetos e programação funcional."
contar = len(re.findall(r'\bPython\b', texto))
print(f"Aparecem no texto {contar} palavras Python")