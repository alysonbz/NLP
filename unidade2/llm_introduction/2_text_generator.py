from transformers import pipeline

llm = pipeline("text-generation")
prompt = "My cat is very"
outputs = llm(prompt, max_length=100)
print(outputs[0]['generated_text'])