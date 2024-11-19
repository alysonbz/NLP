import re

## Primeira questão
'''
def count_word(sentence):
    # Conta as ocorrências da palavra "python" de forma case-insensitive
    count = len(re.findall(r'\bpython\b', sentence, re.IGNORECASE))
    return count

sentence = "Escrevi o código em , python e em python"
result = count_word(sentence)

print(result)
'''
## Segunda questão
'''
def validar_email(email):
    # Expressão regular para validar um endereço de email básico
    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if re.match(regex, email):
        return True
    else:
        return False

email = 'henrique_santos@alu.ufc.br'
resposta = validar_email(email)
print(resposta)
'''
'''
#Quarta Questão
def subistituir_palavra(texto):
    padrao = r'\bgato\b'

    novo_texto = re.sub(padrao, 'cachorro', texto, flags=re.IGNORECASE)
    return novo_texto

texto = ("Odeio gato, sinceramente todas as vezes que vejo gato "
         "tenho vontade de atirar sorvete na cara dos cachorros")
results = subistituir_palavra(texto)
print(results)
'''

'''
## sexta questão
def validar_senha(senha):
  # Expressão regular para verificar se a senha atende aos critérios
  regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'

  # - ^: Início da string
  # - (?=.*[a-z]): Deve conter pelo menos uma letra minúscula
  # - (?=.*[A-Z]): Deve conter pelo menos uma letra maiúscula
  # - (?=.*\d): Deve conter pelo menos um dígito  

  # - (?=.*[@$!%*?&]): Deve conter pelo menos um caractere especial
  # - [A-Za-z\d@$!%*?&]{8,}: Deve ter pelo menos 8 caracteres dos conjuntos anteriores
  # - $: Fim da string

  if re.match(regex, senha):
    return True
  else:
    return False

senha = "Henrique789$"
result = validar_senha(senha)
print(result)
'''

## setima
'''
def extrair_palavras(texto):

  # Expressão regular para encontrar palavras: uma ou mais letras
  padrao = r'\b[a-zA-Z]+\b'

  # Encontrar todas as correspondências do padrão no texto
  palavras = re.findall(padrao, texto)

  return palavras

texto = "Bebi hoje."
results = extrair_palavras(texto)
print(results)
'''

##Oitava Questão
'''
def validar_data(data):

  # Expressão regular para validar a data
  padrao = r"^(0[1-9]|1[0-9]|2[0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$"

  # Verifica se a data corresponde ao padrão
  if re.match(padrao, data):
    return True
  else:
    return False

  texto = "25/10/2002"
  results = validar_data(texto)
  print(results)
'''

## Nona Questão




##Décima questão:
'''
def contar_vogais(texto):
  # Expressão regular para encontrar vogais (a, e, i, o, u)
  padrao = r'[aeiouAEIOU]'

  # Encontrar todas as correspondências do padrão no texto
  vogais = re.findall(padrao, texto)

  # Retornar o número de vogais encontradas
  return len(vogais)

texto = "Comi hoje"
results = contar_vogais(texto)
print(results)
'''

