from transformers import pipeline

# Texto de exemplo
prompt = "A comida foi adequada. O serviço do garçom foi lento."

# Carregar o pipeline para classificação de sentimento
classifier = pipeline('sentiment-analysis')

# Passar a revisão do cliente para o modelo para previsão
prediction = classifier(prompt)

print(prediction)
