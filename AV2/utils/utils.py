from transformers import EvalPrediction
import numpy as np
from sklearn.metrics import accuracy_score
import re

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)  # Remove pontuações
    return text.split()  # Divide em palavras