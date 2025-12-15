from transformers import pipeline

# Load question answering pipeline
qa_pipeline = pipeline("question-answering")

context = """
Machine learning is a subset of artificial intelligence that focuses
on the development of algorithms that allow computers to learn from data.
"""

question = "What is machine learning?"

answer = qa_pipeline(question=question, context=context)

print(answer)