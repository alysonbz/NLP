from transformers import pipeline
llm = pipeline("question-answering")

context = "Eduardo is a data science student at UFC"
question = "Who is Eduardo?"

outputs = llm(question=question, context=context)
print(outputs['answer'])