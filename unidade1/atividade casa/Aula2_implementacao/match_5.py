import re

word_regex = '\s+'
resp = re.split(word_regex, 'semana de aula')
print(resp)

word_regex_2 = r'\!'
resp2 = re.split(word_regex_2, 'semana quente! de aula')
print(resp2)