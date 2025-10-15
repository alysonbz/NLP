import re 

def extrair_nomes_proprios(txt):
    padrão = r"\s?[A-Z][a-zãéáíóú]+\s?"
    return re.findall(pattern=padrão, string=txt)

txt = "Maria Silva João José ana maria"
print(extrair_nomes_proprios(txt))