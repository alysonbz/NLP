# 10. Contagem de vogais:

import re

comp_vogais = re.compile(r"[aeiouAEIOUáàâãéêíóôõúüÁÀÂÃÉÊÍÓÔÕÚÜ]")

texto = "Expressões Regulares em Português"
qtd_vogais = len(comp_vogais.findall(texto))

print("Texto completo:", texto)
print("\nQuantidade de vogais:", qtd_vogais)