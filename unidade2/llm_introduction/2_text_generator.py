from transformers import pipeline

# Carrega o pipeline para geração de texto usando o modelo padrão
gerador_texto = pipeline("text-generation")

# Define o prompt de entrada
prompt = "Ravi lived in Ceará,"

# Gera o texto com base no prompt de entrada
texto_gerado = gerador_texto(prompt, max_length=50, num_return_sequences=1)

# Exibe o texto gerado
print(texto_gerado)
