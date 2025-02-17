from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split


from torch.utils.data import Dataset
import torch

class torch_dataset(Dataset):
    def __init__(self, texts, labels, vocab=None, tokenizer=None, max_len=50):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.tokenizer:
            # Se um tokenizer do Hugging Face for passado, usa ele
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(0),  # Remove batch dim
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

        elif self.vocab:
            tokens = text.lower().split()
            indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
            if len(indices) < self.max_len:
                indices += [self.vocab['<pad>']] * (self.max_len - len(indices))
            else:
                indices = indices[:self.max_len]

            return {
                "input_ids": torch.tensor(indices, dtype=torch.long),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

        else:
            raise ValueError("É necessário fornecer um vocab ou um tokenizer!")




def train_test(csv, test_size=0.2, random_state=42):
    df = pd.read_csv(csv)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv("train_csv.csv", index=False)
    test_df.to_csv("test_csv.csv", index=False)

