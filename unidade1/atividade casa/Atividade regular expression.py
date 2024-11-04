import re

# 1 contagem de correspondencia
def contagem_python(texto):
    return len(re.findall(r'\bPython\b', texto))
print(contagem_python(("Python é complexo, mas o que esperar se Python é a base mais facil da programação.")))

# 2. Validação de email
def validar_email(email):
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))
print(validar_email("abobrinha@hotmail.com"))

# 3. numero de telefone
def numero_telefone(texto):
    return re.findall(r'/b/d{2,3}[-./s]?/d{4,5}[-./s]?/d{4}/b', texto)
print(numero_telefone("call me in 123-4567 ou 98765-4321"))

# 4. substituição de palavras
def substituindo(texto):
    return re.sub(r'\bgato\b', 'cachorro', texto)
print(substituindo("o gato fugiu quando o portão estava aberto"))

# 5. URLs
def extraindo_URLs(texto):
    return re.findall(r'https?://[^\s]+', texto)
print(extraindo_URLs("Visite nosso site https://ufc.com e http://ufc_itapeje.com"))

# 6.  Senha segura
def senha_segura(senha):
    return bool(re.match(r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', senha))
print(senha_segura("NLP_2024"))

# 7. Extração de palavras
def extracao_de_palavras(texto):
    return re.findall(r'\b\w+\b', texto)
print(extracao_de_palavras("Estou enloquecendo nesse ultimo semestre"))

# 8. valisaçãop de data
def validacao_data(data):
    return bool(re.match(r'^\d{2}/\d{2}/\d{4}$', data))
print(validacao_data("04/03/2004"))

# 9. extração de nomes prorpios
def extracao_nomes(texto):
    return re.findall(r'\b[A-Z][a-z]*\b', texto)
print(extracao_nomes("Eryka e seus colegas Joao e Maria estao reprovando em matematica"))

# 10. contagem de vogais
def contagem_vogal(texto):
    return len(re.findall(r'[aeiouAEIOU]', texto))
print(contagem_vogal("Contagem de Vogais nessa frase!"))