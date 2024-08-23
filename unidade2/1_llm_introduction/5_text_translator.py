from transformers import pipeline

llm = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
text = "Walking amid Gion's Machiya wooden houses was a mesmerizing experience."
outputs = llm(text, clean_up_tokenization_spaces=True)
print(outputs[0]['translation_text'])
