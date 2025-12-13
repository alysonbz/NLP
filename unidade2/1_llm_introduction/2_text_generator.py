from transformers import pipeline

generator = pipeline("text-generation")

prompt = "A Inteligência artificial vai mudar o mundo because"

result = generator(prompt, max_length=50, num_return_sequences=1)

print(result[0]["generated_text"])
