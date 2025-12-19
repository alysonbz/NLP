from transformers import pipeline
llm = pipeline("question-answering")

context = "Victor Matheus is a data science student at UFC"
question = "Who is Victor Matheus?"

outputs = llm(question=question, context=context)
print(outputs['answer'])