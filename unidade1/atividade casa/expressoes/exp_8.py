import re

def validar_data(data):
    pattern = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$'
    return bool(re.match(pattern, data))

data = "25/12/2024"
print(validar_data(data))