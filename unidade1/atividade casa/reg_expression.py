#atividade data  04/04

#10 funções para análise de expressões regulares

import re

def contar_palavras(texto, palavra):
    padrao = rf'\b{re.escape(palavra)}\b'
    contagem = len(re.findall(padrao.lower(), texto.lower()))
    return contagem


def validar_email(email):
    padrao = r'^[A-z0-9._%+-]+@[A-z0-9.-]+\.[A-z]{2,}$'

    if re.match(padrao, email):
        return 'e-mail válido'
    else:
        return 'e-mail invalido'


def extrair_numeros_telefone(texto):
    padrao = r'(?:(?:\+\d{1,2}\s?)?\(?\d{2,3}\)?[\s.-]?)?\d{4,5}[\s.-]?\d{4}\b'

    numeros_telefone = re.findall(padrao, texto)

    return numeros_telefone


def substituir_palavra(texto, palavra_antiga, palavra_nova):
    padrao = re.compile(r'\b' + re.escape(palavra_antiga) + r'\b', flags=re.IGNORECASE)

    texto_substituido = padrao.sub(palavra_nova, texto)

    return texto_substituido


def extrair_urls(texto):
    padrao = r'https?://\S+'

    urls = re.findall(padrao, texto)

    return urls


def validar_senha(senha):
    padrao = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'

    if re.match(padrao, senha):
        return 'senha válida'
    else:
        return 'senha inválida'


def extrair_palavras(frase):
    padrao = r'\b\w+\b'

    palavras = re.findall(padrao, frase)

    return palavras


def validar_data(data):
    padrao = r'^\d{2}/\d{2}/\d{4}$'

    if re.match(padrao, data):
        return 'Formato válido'
    else:
        return 'Formato inválido'


def extrair_nomes_proprios(texto):
    padrao = r'\b[A-Z][a-z]+\b'

    nomes_proprios = re.findall(padrao, texto)

    return nomes_proprios


def contar_vogais(texto):
    padrao = r'[aeiouAEIOU]'

    vogais = re.findall(padrao, texto)

    num_vogais = len(vogais)

    return num_vogais



texto = 'Os desenvolvedores usam o Python porque é eficiente e fácil de aprender e pode ser executada em muitas plataformas diferentes.'
palavra = 'python'
resultado = contar_palavras(texto, palavra)
print('Questão 1', '-' * 100, '\n',
      'Texto original:', texto, resultado, '\n',
      'Palavra sendo contada:', palavra, '\n',
      'Saída:', contar_palavras(texto, palavra))


email1 = "brunabarretomq@email.com"
email2 = "tiagodecastro@"
email3 = "VM800916.com"
email4 = 'ruanrodrigues9@ufc.com.br'
print('\nQuestão 2', '-' * 100, '\n',
      'Email:', email1, '\n',
      'Saída:', validar_email(email1), '\n\n',
      'Email:', email2, '\n',
      'Saída:', validar_email(email2), '\n\n',
      'Email:', email3, '\n',
      'Saída:', validar_email(email3), '\n\n',
      'Email:', email4, '\n',
      'Saída:', validar_email(email4), '\n',)


texto = 'Meu número de telefone é (12) 9456-7890.'
numeros = extrair_numeros_telefone(texto)
print('Questão 3', '-' * 100, '\n',
      'Texto:', texto, '\n',
      'Saída:', numeros, '\n')


texto = "Eu tenho um gato preto e um gato branco."
texto_modificado = substituir_palavra(texto, "gato", "cachorro")
print('Questão 4', '-' * 100, '\n',
      'Texto:', texto, '\n',
      'Saída:', texto_modificado, '\n')


texto = 'O site do SIGAA é https://si3.ufc.br/sigaa/verTelaLogin.do e da Minha Biblioteca é https://dliportal.zbra.com.br/Login.aspx?key=UFC'
lista_urls = extrair_urls(texto)
print('Questão 5', '-' * 100, '\n',
      'Texto:', texto, '\n',
      'Saída:', lista_urls, '\n')


senha1 = "Senh@123"
senha2 = "123"
senha3 = "senhafraca"
print('Questão 6', '-' * 100, '\n',
      'Senha:', senha1, '\nSaída:', validar_senha(senha1), '\n',  # Saída: True
      'Senha:', senha2, '\nSaída:', validar_senha(senha2), '\n',  # Saída: False
      'Senha:', senha3, '\nSaída:', validar_senha(senha3), '\n')  # Saída: False


texto = 'Eu tenho um gato preto e um gato branco.'
print('Questão 7', '-' * 100, '\n',
      'Texto:', texto, '\n',
      'Saída:', extrair_palavras(texto), '\n')


data1 = '01/01/2024'
data2 = '2004-06-24'
data3 = '2024/02/22'
print('Questão 8', '-' * 100, '\n',
      'Data:', data1, '\n',
      'Saída:', validar_data(data1), '\n',
      'Data:', data2, '\n',
      'Saída:', validar_data(data2), '\n',
      'Data:', data3, '\n',
      'Saída:', validar_data(data3), '\n')


texto = 'Ana e Alice viajaram para a Argentina.'
print('Questão 9', '-' * 100, '\n',
      'Texto:', texto, '\n',
      'Saída:', extrair_nomes_proprios(texto), '\n')


texto = 'Ana e Alice viajaram para a Argentina.'
print('Questão 10', '-' * 100, '\n',
      'Texto:', texto, '\n',
      'Saída:', contar_vogais(texto))


