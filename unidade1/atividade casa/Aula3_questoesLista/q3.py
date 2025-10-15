# 3. Extração de números de telefone:

import re

numeros = "Telefones: (11) 98765-4321, mais um telefone aqui: 21 3456-7890, +55 31 91234-5678, 11987654321., 22 1234_4321"

tel = re.findall(r"(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?\d{4,5}[\s.-]?\d{4}", numeros)

print(f"Todos os telefones são: {numeros}")
print(f"Telefones encontrados: {tel}")