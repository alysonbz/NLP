import re 

num1 = "85994149930"
num2 = "meu numero é: (85) 99414-9930, mas também pode add ai como (85) 94149930 ou 85994149930 ou (85)99414-9930, mas nao coloca 85941"

def achar_numero(txt):
    # padrão1 = r"\d{11}"
    padrão1 = r"\(?\d{2}\)?\s?\d{4,5}-?\d{4}"
    return re.findall(pattern=padrão1, string=txt) 

print(achar_numero(num2))