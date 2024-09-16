from transformers import pipeline

# Inicializar o pipeline de perguntas e respostas usando o modelo default
question_answerer = pipeline("question-answering")

# Contexto em que a pergunta será respondida
context = """
A inteligência artificial (IA) refere-se a sistemas ou máquinas que imitam a inteligência humana para realizar tarefas e podem se aprimorar iterativamente com base nas informações que coletam. 
Ela está presente em diversos setores, como saúde, educação, indústria, e muitos outros, auxiliando na tomada de decisões e na automação de processos.
"""

# Pergunta
question = "O que a inteligência artificial imita?"

# Obter a resposta com base no contexto fornecido
result = question_answerer(question=question, context=context)

# Exibir a pergunta e a resposta
print(f"Pergunta: {question}")
print(f"Resposta: {result['answer']}")
