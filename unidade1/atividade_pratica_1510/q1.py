import re

def contagem_correspondencia(txt) -> int:
    padrão = r"Python"
    return len(re.findall(pattern=padrão, string=txt))

txt = "Python PythonPython"

print(contagem_correspondencia(txt))