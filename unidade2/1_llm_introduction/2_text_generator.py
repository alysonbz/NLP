from transformers import pipeline

llm = pipeline("text-generation")
prompt = "The Sion neighbohood in Kyoto is famous for"
outputs = llm(prompt, max_length=100)
print(outputs[0]['generated_text'])