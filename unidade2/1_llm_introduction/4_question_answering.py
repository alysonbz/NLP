from torch.distributed.autograd import context
from transformers import pipeline
llm = pipeline("question-answering")
context = "Walking amid Gion's Machiya wooden houses was a mesmerizing experience."
question = "What are Machiya houses made of?"
outputs = llm(question=question, context=context)
print(outputs['answer'])