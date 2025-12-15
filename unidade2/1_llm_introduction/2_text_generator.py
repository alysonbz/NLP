from transformers import pipeline

# Load text generation pipeline (modelo default)
generator = pipeline("text-generation")

prompt = "A educação desempenha um papel importante na sociedade porque"

# Generate text
output = generator(prompt, max_length=100, num_return_sequences=1)

print(output[0]["generated_text"])
