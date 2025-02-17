from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.lower().split()
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        # Padding ou truncamento
        if len(indices) < self.max_len:
            indices += [self.vocab['<pad>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return {
            "input_ids": torch.tensor(indices, dtype=torch.long),
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
    dataset_path = "dataset/b2w_novo.csv"

    train_df, test_df = train_test(dataset_path, test_size=0.2, random_state=42, save=True)

    print("Processo de divisão do dataset concluído.")
