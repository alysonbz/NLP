import re

# Primeira 

def qnt_ocorrencias(ER, string):
    lista = re.findall(ER, string, re.I)
    qnt = len(lista)
    return qnt

string = '''
         Hello python no Python é mais fácil que outras linguagens que não sejam PYTHON,
         PYthon é mais lega.
         A logo do PytHOn são duas cobras descobri isso agora.
         Te amo Phton
         '''

RE = {
    1 : r'python'
}
# Teste
qnt_ocorrencias(ER = r'python', string = string)

# Segunda
'brunabarretomq@gmail.com'

'''
Outlook
gmail.com
gmail.com
hotmail
'''

def valid_email(Args*):
    return None