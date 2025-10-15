# 2. Validação de emails:

import re

test = input("Digite seu email: ")
email = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

if email.match(test):
    print(f"Email encontrado: {test}")
else:
    print("Email invalido")

# Para o teste:
"""
Email válido: Leandroadegas2@gmail.com
Email inválido: Leandroadegas2@@gmail,com.com
"""
