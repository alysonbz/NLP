import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

TOKEN = "<quando rodar, coloque o token>"
model = AutoModelForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2,
                                                           token=TOKEN)
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased", token=TOKEN)


#Carregar dataset de teste
test_df = pd.read_csv("dataset/test.csv")
texts = test_df['tweet_text'].dropna().astype(str).tolist()
labels = test_df['sentiment'].astype(int).tolist()

# Tokenizar os textos
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Fazer previsões
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).numpy()

# Exibir classification report
print("Classification Report:")
print(classification_report(labels, predictions, target_names=["Negativo", "Positivo"]))

# Gerar e exibir matriz de confusão
matriz_conf = confusion_matrix(labels, predictions)
ConfusionMatrixDisplay(matriz_conf, display_labels=["Negativo", "Positivo"]).plot()
plt.title("Matriz de Confusão - Modelo Pré-Treinado")
plt.show()