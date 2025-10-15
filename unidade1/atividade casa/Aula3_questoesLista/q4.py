# 4. Substituição de palavras:

import re

texto = "Eu gosto de tocar teclado."
print(f"Texto antes de substituir: {texto}")

subs = re.sub(r"\bteclado\b", "guitarra", texto)
print(f"Texto depois de substituir: {subs}")
