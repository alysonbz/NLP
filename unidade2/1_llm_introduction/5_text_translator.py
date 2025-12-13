from transformers import pipeline

translator = pipeline("translation")

text = "Artificial intelligence is transforming the world."

translation = translator(text)

print(translation[0]["translation_text"])
