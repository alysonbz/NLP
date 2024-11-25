import re

#atividade data  04/04

#10 funções para análise de expressões regulares

#1.
text1 = "Eu amo estudar python, pois python é fácil"
fill1 = len(re.findall(r'\bpython\b', text1, flags=re.IGNORECASE))  # "r" serve para a palavra ser identifado como
# uma string bruta
print(fill1)

#2.
text2 = "erickcoutinho@gmail.com"
if not re.search("@", text2):
    print("O texto não é um email válido")
else:
    print("O texto é um email válido")

#3.
text3 = "Esse é meu número de telefone: 991293270"
fill3 = re.findall(r'\b\d{9}\b', text3)
print(fill3)

#4.
text4 = "Eu meu gato"
fill4 = re.sub(r'\bgato\b', 'cachorro', text4, flags=re.IGNORECASE)
print(fill4)

#5.
text5 = "link da minha musica: https://www.youtube.com/watch?v=yYvb7KzkDpY&list=RDyYvb7KzkDpY&start_radio=1"
fill5 = re.findall(r'https:[^\s]+', text5)
print(fill5)

#6.
text6 = "erick123"

#7.
text7 = "Gosto de filmes de aventura"
fill7 = re.findall(r'\b\w+\b', text7)
print(fill7)

#8.
text8 = "2024/11/04"

#9.
text9 = "O nome dele é Carlos"
fill9 = re.findall(r'\b[A-Z][a-zA-Z]*\b', text9)
print(fill9)

#10.
text10 = "Eu amo viajar"
fill10 = len(re.findall(r'[aeiouAEIOU]', text10))
print(fill10)