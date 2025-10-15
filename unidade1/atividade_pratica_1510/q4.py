import re 

def trocar_gato_por_cachorro(txt):
    padrão = r"gato"
    return re.sub(pattern=padrão, repl="cachorro", string=txt)

txt = "O gato é um animal que é muito legal. Tenhho um gato"
print(trocar_gato_por_cachorro(txt))