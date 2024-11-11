# 4) Uma função que retorne a quantidade de palavras que estão  em caixa alta.
import re
def caixa_alta(texto):
    retorne = re.findall(r'[A-Z][A-Z]*',texto)
    return len(retorne)

texto = "O QUE É QUE CACÁ QUER? CACÁ QUER CAQUI. QUAL CAQUI QUE CACÁ QUER? CACÁ QUER QUALQUER CAQUI"
contar = caixa_alta(texto)
print(contar)