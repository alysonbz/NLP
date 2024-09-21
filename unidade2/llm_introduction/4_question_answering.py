from transformers import pipeline

llm = pipeline("question-answering")
context = ("""A menina pegou o brinquedo do irnão por achar o brinquedo mais legal do que o que ela havia ganhado""")

question = " Por que a a menina pegou o brinquedo do irmão?"
outputs = llm(question=question, context=context)
print(outputs['answer'])