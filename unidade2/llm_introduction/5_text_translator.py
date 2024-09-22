from transformers import pipeline

# Carrega o pipeline para tradução
tradutor = pipeline("translation_en_to_fr")

# Define o texto de entrada para tradução
texto = "Ravi lived in Itapaje."

# Usa o modelo para traduzir o texto
traducao = tradutor(texto)

# Exibe o texto traduzido
print(traducao)