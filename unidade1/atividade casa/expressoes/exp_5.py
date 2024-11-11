import re

def extrair(texto):
    pattern = r'https?://(?:ww\.)?\S+'
    return re.findall(pattern, texto)

texto = "Confira o site http://exemplo.com e também https://www.site.com.br para mais informações."
print(extrair(texto))