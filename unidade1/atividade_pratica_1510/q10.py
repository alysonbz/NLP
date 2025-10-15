import re 

def contar_vogais(txt):
    padrão = r"[aeiou]"
    return len(re.findall(pattern=padrão, string=txt))

txt = "O gato é um animal que é muito legal. Tenho um gato"
print(contar_vogais(txt))

print(contar_vogais('aeiousla'))