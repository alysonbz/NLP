# 9. Extração de nomes próprios:

import re

texto = "Ana e João foram a São Paulo. José da Silva encontrou Maria. brasil não entra."
nomes = re.findall(r"\b[A-ZÁÀÂÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç]+\b", texto)

print(f"Texto completo: {texto}")
print(f"\nNomes próprios: {nomes}")