# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.
import re

def numerico(texto):
    retorno =re.findall(r'd+',texto)
    return len(retorno)

texto = "1 elefante incomoda muita gente, 2 elefantes incomodam muito mais, 3 elefantes incomodam muita gente, 4 elefantes incomodam muito mais"
contar = numerico(texto)
print(contar)