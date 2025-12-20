from transformers import pipeline
llm = pipeline("translation_en_to_fr")

text = "Eduardo is a data science student at UFC"

outputs = llm(text)
print(outputs[0]['translation_text'])