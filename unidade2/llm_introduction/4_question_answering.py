from transformers import pipeline

# Inicializa o pipeline de resposta a perguntas
qa_pipeline = pipeline("question-answering")

# Texto de contexto
context = """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
It has become an essential part of the technology industry. Research associated with AI is highly technical and specialized.
The core problems of AI include programming computers for certain traits such as knowledge, reasoning, problem-solving,
perception, learning, planning, and the ability to manipulate and move objects.
"""

# Pergunta a ser respondida com base no contexto
question = "What are the core problems of AI?"

# Realiza a resposta à pergunta
answer = qa_pipeline(question=question, context=context)

# Exibe a resposta gerada
print(f"Question: {question}")
print(f"Answer: {answer['answer']}")