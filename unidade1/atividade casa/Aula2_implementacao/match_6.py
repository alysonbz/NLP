import re

word_regex = r'[a-z]'
resp = re.split(word_regex, 'Semana Quente! De Aula')
print(resp)

word_regex_2 = r'[a-z]\w+'
resp2 = re.split(word_regex_2, '4 Semanas Quente! De Aula')
print(resp2)

word_regex_3 = r'[a-z]\w+'
resp3 = re.findall(word_regex_3, '4 Semanas Quente! De Aula')
print(resp3)
