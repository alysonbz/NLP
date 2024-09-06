#No arquivo 1_sentiment_classifier.py  elabore uma classificação de sentimento
# usando um modelo pré-treinado BERT.
from transformers import pipeline

prompt = "A comida foi adequada. O serviço do garçom foi lento."

# Load the pipeline for sentiment classification
classifier = pipeline(task="text-classification",
                      model="nlptown/bert-base-multilingual-uncased-sentiment")

# Pass the customer review to the model for prediction
prediction = classifier(prompt)

print(prediction)
