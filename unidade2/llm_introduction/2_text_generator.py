from transformers import pipeline

# Inicializa o pipeline de geração de texto com o modelo específico
generator = pipeline("text-generation", model="gpt2")

# Define o prompt inicial para a geração de texto
prompt = "Once upon a time"

# Gera o texto com o modelo, ativando truncamento explícito
generated_text = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)

# Exibe o texto gerado
print(generated_text[0]['generated_text'])