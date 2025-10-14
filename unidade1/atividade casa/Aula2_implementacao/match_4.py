import re

word_regex = '\w+'
resp = re.match(word_regex, 'semana de aula')
print(resp)

resp_2 = re.match(word_regex, 's emana de aula')
print(resp_2)

word_regex_2 = '\d'
resp_3 = re.match(word_regex_2, '42semana de aula')
print(resp_3)

word_regex_3 = '\d+'
resp_4 = re.match(word_regex_3, '42semana de aula')
print(resp_4)


padrao = r'Universidade'
texto = 'universidade federal do ceara'

resultado = re.match(padrao, texto)
print(resultado)

if resultado:
    print('Padrão encontrado no início da string.')
else:
    print('Padrão não encontrado no início da string.')

