from transformers import pipeline

# Texto de entrada
prompt = "A comida foi adequada. O serviço do garçom foi lento."

# Load the pipeline for sentiment classification (BERT pré-treinado)
classifier = pipeline("sentiment-analysis")

# Pass the customer review to the model for prediction
prediction = classifier(prompt)

print(prediction)
