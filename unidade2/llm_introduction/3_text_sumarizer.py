from transformers import pipeline

# Criação do pipeline para sumarização
llm = pipeline("summarization", model="facebook/bart-large-cnn")

# Texto longo de entrada
long_text = """Walking amid Gion's Machiya wooden houses is a mesmerizing experience. The beautifully preserved structures exuded an old-world charm that transports visitors back in time, making them feel like they had stepped into a living museum. The glow of lanterns lining the narrow streets add to the enchanting ambiance, making each stroll a memorable journey through Japan's rich cultural history."""

# Geração do resumo com limite de 60 tokens
outputs = llm(long_text, max_length=60, clean_up_tokenization_spaces=True)

# Impressão do texto resumido
print(outputs[0]['summary_text'])
