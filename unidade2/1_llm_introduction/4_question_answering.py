from transformers import pipeline
llm = pipeline("question-answering")

context = "José Mário is a data science student at UFC"
question = "Who is José Mário?"

outputs = llm(question=question, context=context)
print(outputs['answer'])
