from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "I was at the park and"

generated_text = generator(prompt,
                           max_length=512,
                           num_return_sequences=1,
                           truncation=True)

print(generated_text[0]['generated_text'])
