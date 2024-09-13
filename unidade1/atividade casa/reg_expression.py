import re

# 1. Contagem de Correspondências
def contar_ocorrencias_python(texto):
    return len(re.findall(r'Python', texto))

# 2. Validação de E-mail
def validar_email(email):
    padrao = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(padrao, email) is not None

# 3. Extração de Números de Telefone
def extrair_numeros_telefone(texto):
    padrao = r'\+?\d[\d -]{8,}\d'
    return re.findall(padrao, texto)

# 4. Substituição de Palavras
def substituir_palavra(texto):
    return re.sub(r'\bfilho\b', 'Ravi', texto)


# 5. Extração de URLs
def extrair_urls(texto):
    padrao = r'https?://[^\s/$.?#].[^\s]*'
    return re.findall(padrao, texto)

# 6. Verificação de Senha Segura
def verificar_senha_segura(senha):
    padrao = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return re.match(padrao, senha) is not None

# 7. Extração de Palavras
def extrair_palavras(texto):
    return re.findall(r'\b\w+\b', texto)

# 8. Validação de Data
def validar_data(data):
    padrao = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/([0-9]{4})$'
    return re.match(padrao, data) is not None

# 9. Extração de Nomes Próprios
def extrair_nomes_proprios(texto):
    return re.findall(r'\b[A-Z][a-z]*\b', texto)

# 10. Contagem de Vogais
def contar_vogais(texto):
    return len(re.findall(r'[aeiouAEIOU]', texto))

# Função principal para demonstrar as funcionalidades
def main():
    # Exemplos de uso para cada função
    texto = "Python é a linguagem de programação que mais utilizo. Já ministrei um minicurso básico de Python."
    email = "lauramendes@alu.ufc.br"
    texto_telefone = "Meu número de telefone é +55 31 99873-8051."
    texto_substituicao = "Meu filho é recém nascido. Meu filho tem 20 dias"
    texto_urls = "A atividade está em https://github.com/alysonbz/NLP/tree/main"
    senha = "SenhaAtividade12345!"
    texto_palavras = "Python é uma linguagem de programação."
    data = "13/09/2024"
    texto_nomes_proprios = "Laura e Ravi foram ao médico com o uber Pedro."
    texto_vogais = "Laura e Ravi foram ao médico com o uber Pedro."

    print("Contagem de 'Python':", contar_ocorrencias_python(texto))
    print("E-mail :", validar_email(email))
    print("Número de telefone:", extrair_numeros_telefone(texto_telefone))
    print("Texto após substituição de 'filho' por 'Ravi':", substituir_palavra(texto_substituicao))
    print("URLs extraída:", extrair_urls(texto_urls))
    print("Senha segura:", verificar_senha_segura(senha))
    print("Palavras extraídas:", extrair_palavras(texto_palavras))
    print("Data válida:", validar_data(data))
    print("Nomes próprios extraídos:", extrair_nomes_proprios(texto_nomes_proprios))
    print("Contagem de vogais:", contar_vogais(texto_vogais))

if __name__ == "__main__":
    main()
