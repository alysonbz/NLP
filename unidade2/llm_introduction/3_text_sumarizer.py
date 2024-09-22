from transformers import pipeline

# Carrega o pipeline para sumarização de texto usando o modelo padrão
summarizer = pipeline("summarization")

# Define o texto de entrada para sumarização
texto = """
A gravidez é um periodo unico e transformador na vida de uma mulher, marcado por inumeras mudanças físicas, emocionais e psicológicas. 
Durante os nove meses de gestação, o corpo da mae passa por adaptaçoes complexas para nutrir e proteger o bebê em desenvolvimento.
"""

# Gera o sumário do texto
resumo = summarizer(texto, max_length=60, min_length=20, do_sample=False)

# Exibe o resumo gerado
print(resumo)