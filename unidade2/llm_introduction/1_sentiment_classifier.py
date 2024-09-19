from transformers import pipeline

# Load the pipeline for sentiment classification using a pre-trained BERT model
classifier = pipeline('sentiment-analysis', model='bert-base-uncased')

# Define a prompt
prompt = "A comida foi adequada. O serviço do garçom foi lento."

# Pass the customer review to the model for prediction
prediction = classifier(prompt)

print(prediction)
