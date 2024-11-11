import re

def validar_email(email):
    validar = r'^[a-zA-Z0-9_.]+@[a-zA-Z0-9_.]+\.[a-zA-Z0-9_.]+$'
    return bool(re.match(validar, email))

email = input('Digite seu email:')
if validar_email(email):
    print('Email aceito!')
else:
    print('Email não aceito!')
