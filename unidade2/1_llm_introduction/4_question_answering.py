from transformers import pipeline

# Load the question answering pipeline (modelo default)
qa_pipeline = pipeline("question-answering")

context = "The Amazon rainforest is the largest tropical rainforest in the world. It plays a crucial role in regulating the Earth's climate."

question = "What is the role of the Amazon rainforest?"

answer = qa_pipeline(question=question, context=context)
print(answer["answer"])
