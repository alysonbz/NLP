# 6. Verificação de senha segura:

import re

senhas = ["Aa1@aaaa", "senha123", "ABCdef!!", "Aa1@", "BomDia#2024"]

for s in senhas:
    if (
        len(s) >= 8       # definindo o tamanho da senha para 8
        and re.search(r'[a-z]', s)
        and re.search(r'[A-Z]', s)
        and re.search(r'\d', s)
        and re.search(r'[^A-Za-z0-9]', s)
    ):
        print(f"As senhas válidas são: {s}")