import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from preprocessing import torch_dataset
from utils.utils import preprocess_text
from sklearn.metrics import accuracy_score

class CustomModelTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=2):
        super(CustomModelTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# Criando vocabulário
train_df = pd.read_csv("subs/df_train.csv")
texts = train_df['text'].dropna().tolist()

def tokenize(text):
    text = text.lower().replace(",", "").replace(".", "")
    tokens = text.split()
    return tokens

all_tokens = set(token for sentence in texts for token in tokenize(sentence))
vocab = {word: idx + 1 for idx, word in enumerate(all_tokens)}
vocab["<PAD>"] = 0
vocab["<UNK>"] = len(vocab)

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Modelo
model = CustomModelTextClassifier(vocab_size=len(vocab)).to(device)

# Definir otimizador e função de perda
optimizer = optim.Adam(model.parameters(), lr=4e-3)
loss_fn = nn.CrossEntropyLoss()

# Criar DataLoader
batch_size = 16
train_dataset = torch_dataset(train_df['text'].dropna().tolist(), train_df['is_hate_speech'].tolist(), vocab=vocab)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Treinamento do modelo do zero
epochs = 50
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in train_dataloader:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calcular previsões e armazenar para acurácia
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Calcular acurácia
    acc = accuracy_score(all_labels, all_preds)

    # Ajustar learning rate com o scheduler **depois do cálculo da métrica**
    scheduler.step(total_loss)

    print(f"Época {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Acurácia: {acc:.4f}")

print("Treinamento concluído!")

# Salvar modelo treinado
torch.save(model.state_dict(), "custom_text_classifier.pth")
print("Modelo salvo como custom_text_classifier.pth")
