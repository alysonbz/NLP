from transformers import pipeline


translator = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")

text = "We are going to Mexico tomorrow."

translation = translator(text, max_length=40, src_lang="en_XX", tgt_lang="pt_BR")

print(f"Original Text: {text}")
print(f"Translated Text: {translation[0]['translation_text']}")