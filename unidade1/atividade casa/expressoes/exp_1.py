import re
def contar(texto):
    ocorre = re.findall(r'\bpython\b', texto, re.IGNORECASE) #considera letras maiusculas e minusculas iguais
    return len(ocorre) #ler a quantidade de vezes que aparece 'python'

texto = 'Python, você é lindo igual um limão azedo feito suco, ai python python python, vc é tão pythonildo'
contagem = contar(texto)

print(f'Quantas vezes aparece a palavra python na frase:\n({texto})?\n{contagem}')