from transformers import pipeline

# Inicializar o pipeline de tradução do inglês para o espanhol
llm = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

# Novo texto em inglês que será traduzido
text = "Exploring the ancient ruins of the city offered a glimpse into its rich history."

# Traduzir o texto para o espanhol
outputs = llm(text, max_length=50)

# Exibir o texto traduzido
print(outputs[0]['translation_text'])
