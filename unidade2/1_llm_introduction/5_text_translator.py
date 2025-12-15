from transformers import pipeline

# Load translation pipeline (default: inglês → francês, pode variar)
translator = pipeline("translation")

text = "Machine learning is changing the world."

translated_text = translator(text)

print(translated_text[0]["translation_text"])
