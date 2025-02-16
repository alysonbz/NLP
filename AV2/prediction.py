from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from AV2.preprocessing import SentimentDataset
import pandas as pd

model_path = "fine_tuning/bert-base-uncased_model_fine_turning_train"
tokenizer_path = "fine_tuning/bert-base-uncased_tokenizer_fine_turning_tokenizer"

# Carregar modelo treinado e tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
test_df = pd.read_csv("dataset/test.csv")

# Criar dataset corretamente
test_dataset = SentimentDataset(
    test_df["tweet_text"].tolist(),
    test_df["sentiment"].tolist(),
    tokenizer=tokenizer
)
# Predições e avaliação
trainer = Trainer(model=model, tokenizer=tokenizer)
preds_output = trainer.predict(test_dataset)
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = np.array([item["labels"].item() for item in test_dataset])


# Relatório de classificação
report = classification_report(y_true, y_pred)
print("Classification Report\n", report)

# Matriz de Confusão
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.title("Matriz de Confusão")
plt.show()