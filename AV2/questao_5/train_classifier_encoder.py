import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from AV2.questao_2.preprocessing import torch_dataset
from collections import Counter
from my_model import TextEncoder

# Configurações
MAX_LEN = 30
VOCAB_SIZE = 5000
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.0001

# CSV
train_df = pd.read_csv("../subs/df_train_balanced.csv")
test_df = pd.read_csv("../subs/df_test_balanced.csv")

# rotulos
train_df['is_hate_speech'] = train_df['is_hate_speech'].astype(int)
test_df['is_hate_speech'] = test_df['is_hate_speech'].astype(int)
NUM_CLASSES = 2

# VOCAB
all_texts = train_df['text'].tolist() + test_df['text'].tolist()
word_counts = Counter(" ".join(all_texts).lower().split())

# mais frequentes
most_common_words = [word for word, _ in word_counts.most_common(VOCAB_SIZE)]
vocab = {word: idx for idx, word in enumerate(most_common_words)}
vocab["<unk>"] = VOCAB_SIZE  # Token para palavras desconhecidas
vocab["<pad>"] = VOCAB_SIZE + 1  # Token de padding

# vocabulário
"""vocab_path = "../models/modelo_do_0/vocab.json"
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)"""

# dataset Torch
train_dataset = torch_dataset(train_df['text'].tolist(), train_df['is_hate_speech'].tolist(), max_len=MAX_LEN,
                              vocab=vocab)
test_dataset = torch_dataset(test_df['text'].tolist(), test_df['is_hate_speech'].tolist(), max_len=MAX_LEN,
                             vocab=vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# instanciar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextEncoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Early Stopping
early_stop_patience = 20
best_val_loss = float("inf")
epochs_without_improvement = 0

# treinamento
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Avaliação
    model.eval()
    val_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    accuracy = sum(1 for x, y in zip(all_labels, all_preds) if x == y) / len(all_labels)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model, "../models/modelo_do_0/text_encoder_my_model.pt")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stop_patience:
            break

# salvar modelo
model_path = "text_encoder_my_model_weights_2.pt"
torch.save(model.state_dict(), model_path)
