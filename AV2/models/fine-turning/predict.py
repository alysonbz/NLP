from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from AV2.preprocessing import torch_dataset
import pandas as pd

model_path = "gold_standard/neuralmind_bert-large-portuguese-cased_model_fine_tuning_data_balanced"
tokenizer_path = "gold_standard/neuralmind_bert-large-portuguese-cased_tokenizer_fine_tuning_data_balanced"

# Carregar modelo treinado e tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
test_df = pd.read_csv("../../subs/test_csv_balanced.csv")

# Criar dataset corretamente
test_dataset = torch_dataset(
    test_df["text"].tolist(),
    test_df["is_hate_speech"].tolist(),
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
