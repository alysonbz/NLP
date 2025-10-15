import re

def extrair_palavras(txt):
    padrão = r"\w+"
    return re.findall(pattern=padrão, string=txt)

txt = "O gato é um animal que é muito legal. Tenhho um gato"
print(extrair_palavras(txt))