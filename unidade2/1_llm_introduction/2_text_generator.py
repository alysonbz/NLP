from transformers import pipeline
# Load the pipeline for text generation
model = pipeline("text-generation")

prompt = "Victor Matheus is "

outputs = model(prompt, max_length=100)
print(outputs[0]['generated_text'])