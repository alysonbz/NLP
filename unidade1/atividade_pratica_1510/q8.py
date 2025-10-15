import re

def extrair_data(txt):
    padrão = r"\d{2}/\d{2}/\d{4}"
    return re.findall(pattern=padrão, string=txt)

txt = "Data1: 12/12/2020, data2: 12-12-2020, data3: 12/12-2020"
print(extrair_data(txt))