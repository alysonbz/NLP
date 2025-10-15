# 7. Extração de palavras:

import re

texto = "Expressões regulares em Português: Rápido, fácil e útil!"
palavras_val = re.findall(r"[A-Za-zÀ-Öà-öø-ÿ0-9_]+", texto)

print(f"Texto a ser extraído: {texto}")
print("\nExtração das palavras:", palavras_val)