from transformers import pipeline

# Inicializa o pipeline de tradução (inglês para francês)
translator = pipeline("translation_en_to_fr")

# Texto a ser traduzido
text = "Artificial Intelligence is revolutionizing the world of technology."

# Realiza a tradução do texto
translation = translator(text, max_length=40)

# Exibe o texto traduzido
print(f"Original Text: {text}")
print(f"Translated Text: {translation[0]['translation_text']}")