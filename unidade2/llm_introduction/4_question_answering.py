from transformers import pipeline

# Criação do pipeline para resposta a perguntas
llm = pipeline("question-answering")

# Contexto do texto
context = "Walking amid Gion's Machiya wooden houses was a mesmerizing experience."

# Pergunta
question = "What are Machiya houses made of?"

# Obtenção da resposta com base no contexto
outputs = llm(question=question, context=context)

# Impressão da resposta
print(outputs['answer'])
