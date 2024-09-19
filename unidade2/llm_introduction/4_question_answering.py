from transformers import pipeline

# Carrega o pipeline para responder perguntas usando o modelo padrão
resposta_pergunta = pipeline("question-answering")

# Define o contexto e a pergunta
contexto = """
A gravidez é um período único e transformador na vida de uma mulher, 
marcado por inúmeras mudanças físicas, emocionais e psicológicas. 
Durante os nove meses de gestação, o corpo da mãe passa por adaptações complexas 
para nutrir e proteger o bebê em desenvolvimento.
"""

pergunta = "Quais são as mudanças que o corpo da mãe passa durante a gravidez?"

# Usa o modelo para responder a pergunta com base no contexto
resposta = resposta_pergunta(question=pergunta, context=contexto)

# Exibe a resposta
print(resposta)
