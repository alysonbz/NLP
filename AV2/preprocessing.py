import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenização usando o tokenizer do BERT
        encoding = self.tokenizer(
            text,
            padding="max_length",  # Preenchimento para comprimento fixo
            truncation=True,       # Trunca textos muito longos
            max_length=self.max_len,
            return_tensors="pt"    # Retorna tensores do PyTorch
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # Remove dimensão extra
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_test(csv, test_size=0.2, random_state=42, save=True):
    df = pd.read_csv(csv)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    if save:
        os.makedirs("dataset", exist_ok=True)
        train_df.to_csv("dataset/train.csv", index=False)
        test_df.to_csv("dataset/test.csv", index=False)
    return train_df, test_df

if __name__ == "__main__":
    dataset_path = "dataset/twitter_sentiment.csv"

    train_df, test_df = train_test(dataset_path, test_size=0.2, random_state=42, save=True)

    print("Processo de divisão do dataset concluído.")
