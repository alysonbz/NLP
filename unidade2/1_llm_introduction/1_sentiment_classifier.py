# Import the function for loading Hugging Face pipelines
from transformers import pipeline

prompt = "The food was good, but service at the restaurant was a bit slow"

# Load the pipeline for sentiment classification
classifier = pipeline("text-classification",'nlptown/bert-base-multilingual-uncased-sentiment')

# Pass the customer review to the model for prediction
prediction = classifier(prompt)

print(prediction)
