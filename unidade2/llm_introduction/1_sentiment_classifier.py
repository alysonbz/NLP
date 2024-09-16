from transformers import pipeline

# Inicializar o pipeline de classificação de sentimento usando o modelo BERT pré-treinado
sentiment_classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Texto a ser classificado
text = "Eu estou muito feliz com o resultado desse projeto!"

# Classificar o sentimento do texto
result = sentiment_classifier(text)

# Exibir o resultado da classificação
print(f"Texto: {text}")
print(f"Classificação: {result[0]['label']}, Score: {result[0]['score']:.4f}")
