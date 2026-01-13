import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from AV1.preprocessing import carregar_dataset

#======================================================
# CONFIGURAÇÕES
#======================================================
MODEL_NAME = "gpt-4o-mini"   # rápido e mais barato
TEMPERATURE = 0
MAX_TEXT_SIZE = 5000
SAMPLE_SIZE = 5000

#======================================================
# OPENAI
#======================================================
load_dotenv()
client = OpenAI()

#======================================================
# FUNÇÃO DE AVALIAÇÃO
#======================================================
def avaliar(y_true, y_pred):
    return {
        "Acuracia": accuracy_score(y_true, y_pred),
        "Precisao": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1": f1_score(y_true, y_pred, average="macro"),
    }

#======================================================
# FUNÇÃO DE CLASSIFICAÇÃO COM OPENAI
#======================================================
def classificar_openai(texto):
    prompt = f"""
Classifique o sentimento do texto abaixo como:
0 = negativo
1 = positivo

Responda APENAS com 0 ou 1.

Texto:
\"\"\"{texto}\"\"\"
"""
    try:
        resposta = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Você é um classificador de sentimentos."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE
        )

        saida = resposta.choices[0].message.content.strip()

        if saida not in ["0", "1"]:
            return None

        return int(saida)

    except Exception as e:
        print("Erro:", e)
        return None


#======================================================
# DATASET
#======================================================
print("\n--- Dataset ---")
df = carregar_dataset("AV1/b2w.csv")
df = df.dropna(subset=["polarity"])
df = df.sample(SAMPLE_SIZE, random_state=42)

coluna_texto = "review_text"
coluna_label = "polarity"

X_train, X_test, y_train, y_test = train_test_split(
    df[coluna_texto], df[coluna_label],
    test_size=0.2, random_state=42
)

#======================================================
# CLASSIFICAÇÃO
#======================================================
print("\n--- LLM OpenAI: Classificação de Sentimento ---")

y_true = []
y_pred = []

for texto, label in zip(X_test, y_test):
    pred = classificar_openai(texto[:MAX_TEXT_SIZE])
    if pred is None:
        continue

    y_pred.append(pred)
    y_true.append(label)

print(f"\nTotal de amostras avaliadas: {len(y_true)}")

#======================================================
# MÉTRICAS
#======================================================
metricas = avaliar(y_true, y_pred)
print("\nResultados OpenAI LLM:")
for k, v in metricas.items():
    print(f"{k}: {v:.4f}")

#======================================================
# MATRIZ DE CONFUSÃO
#======================================================
cm = confusion_matrix(y_true, y_pred)
labels = ["Negativo", "Positivo"]

print("\nMatriz de Confusão:")
print(cm)

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.xticks([0, 1], labels)
plt.yticks([0, 1], labels)
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.title("Matriz de Confusão - OpenAI LLM")
plt.colorbar()
plt.show()
