import torch
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


def tokenize_torch(text, vocab, max_length=20):
    tokens = text.lower().split()
    token_ids = [vocab.get(word, vocab["<unk>"]) for word in tokens]
    token_ids += [vocab["<pad>"]] * (max_length - len(token_ids))
    return torch.tensor([token_ids])
