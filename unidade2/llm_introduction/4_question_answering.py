# No arquivo 4_question_answering.py elabore um programa para
# responder perguntas utilizando o modelo
# default do módulo pipeline
from transformers import pipeline

llm = pipeline("question-answering")
context = ("""A fada do dente 
entrou na casa para pegar o dente da criança""")

question = " Por que a fada do dente entrou na casa?"
outputs = llm(question=question, context=context)
print(outputs['answer'])