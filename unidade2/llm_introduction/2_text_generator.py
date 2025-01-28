from transformers import pipeline

# Criação do pipeline para geração de texto
llm = pipeline("text-generation")

# Prompt de entrada
prompt = "The Gion neighborhood in Kyoto is famous for"

# Geração do texto com limite de 100 tokens
outputs = llm(prompt, max_length=100)

# Impressão do texto gerado
print(outputs[0]['generated_text'])
