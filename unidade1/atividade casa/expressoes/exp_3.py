# ^\(\d{2}\) \d{5} - \d{4}$

import re

def telefone(numero):
    validar = r'\(\d{2}\)\s?\d{5}-\d{4}|\(\d{2}\)\s?\d{4}-\d{4}'
    return re.findall(validar, numero )

numero = "Para mais informações, entre em contato pelos números: (11)95875-7851, (11)91234-5678 e (11)92597-5574. Você também pode ligar para o escritório em (21)1234-5678."
contagem = telefone(numero)

print(contagem)