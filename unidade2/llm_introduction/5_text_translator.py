# No arquivo 5_text_translator.py elabore um programa para traduzir textos,
# utilizando o modelo default do módulo pipeline
from transformers import pipeline

llm = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
text = "Lucy in the sky with diamonds"
outputs = llm(text, clean_up_tokenization_spaces=True)
print(outputs[0]['translation_text'])

