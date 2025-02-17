import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

TOKEN = "token aqui"
model = AutoModelForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2,
                                                           token=TOKEN)
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased", token=TOKEN)


# Dataset test
test_df = pd.read_csv("dataset/test.csv")

texts = test_df['review_text'].dropna().astype(str).tolist()
labels = test_df['rating'].astype(int).tolist()

# Tokenizer texto de teste
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).numpy()

print("Classification Report:")
print(classification_report(labels, predictions))

# Matriz de confusão
matriz_conf = confusion_matrix(labels, predictions)
ConfusionMatrixDisplay(matriz_conf, display_labels=[0, 1]).plot()
plt.show()
