# Import the function for loading Hugging Face pipelines
from transformers import pipeline

prompt = "Maria died after"

llm = pipeline(task="text-generation",
               model="openai-community/gpt2")

#print(llm.model.name_or_path)  # Mostra o modelo usado

# Output
generation = llm(prompt)

print(generation[0]["generated_text"])