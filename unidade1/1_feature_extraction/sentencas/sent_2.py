# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
import re
def palavra(texto):
    retorne = re.findall(r'[A-Z][a-zA-Z]*',texto)
    return len(retorne)

texto = "O que é que Cacá quer? Cacá quer caqui. Qual caqui que Cacá quer? Cacá quer qualquer caqui"
contar = palavra(texto)
print(contar)