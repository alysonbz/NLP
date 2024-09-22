import pandas as pd
import re
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Download de recursos NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# =======================
# Funções de Pré-processamento
# =======================
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Zà-úÀ-Ú]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('portuguese'))
    words = nltk.word_tokenize(text)
    words_filtered = [word for word in words if word not in stop_words]
    return ' '.join(words_filtered)

def stem_text(text):
    stemmer = RSLPStemmer()
    words = nltk.word_tokenize(text)
    words_stemmed = [stemmer.stem(word) for word in words]
    return ' '.join(words_stemmed)

# =======================
# Carregar e Processar Conjunto de Dados
# =======================
filepath = r"/content/complete_dataset.csv"
df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')

# Combinar mensagens verdadeiras e falsas
df_true = df[['message_true']].copy().dropna()
df_true['type'] = 'true'
df_true['message'] = df_true['message_true']

df_false = df[['message', 'type']].copy()
df_false = df_false[df_false['type'] != 'true']  # Filtrar apenas mensagens falsas

df_combined = pd.concat([df_false[['message', 'type']], df_true[['message', 'type']]], ignore_index=True)
df_combined['message'] = df_combined['message'].fillna('')

# Aplicar pré-processamento
df_combined['message'] = df_combined['message'].apply(preprocess_text)
df_combined['message'] = df_combined['message'].apply(remove_stopwords)
df_combined['message'] = df_combined['message'].apply(stem_text)

# Exibir as primeiras linhas do dataframe
print(df_combined.head())

# =======================
# Construir Vocabulário
# =======================
def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        words = sentence.split()
        counter.update(words)
    vocab = {word: idx+2 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

vocab = build_vocab(df_combined['message'].values)
vocab_size = len(vocab)
print(f"Tamanho do vocabulário: {vocab_size}")

# =======================
# Definir Modelo Transformer
# =======================
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, nhead=2, num_encoder_layers=2, dim_feedforward=512):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward), num_encoder_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Average over the sequence length
        x = self.fc(x)
        return x

# =======================
# Classe TextDataset
# =======================
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def encode_text(self, text):
        tokens = text.split()
        encoded = [self.vocab.get(word, self.vocab['<UNK>']) for word in tokens]
        if len(encoded) < self.max_len:
            encoded += [self.vocab['<PAD>']] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
        return encoded

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded_text = self.encode_text(text)
        return torch.tensor(encoded_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

label_mapping = {'true': 0, 'fake': 1}
df_combined['label'] = df_combined['type'].map(label_mapping)

# =======================
# Divisão em Treino e Teste
# =======================
X_train, X_test, y_train, y_test = train_test_split(df_combined['message'], df_combined['label'], test_size=0.2, random_state=42, stratify=df_combined['label'])

# Criar datasets e DataLoaders
train_dataset = TextDataset(X_train.values, y_train.values, vocab)
test_dataset = TextDataset(X_test.values, y_test.values, vocab)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# =======================
# Configuração do Modelo
# =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando o dispositivo: {device}")

model = TransformerClassifier(vocab_size=vocab_size, num_classes=len(label_mapping)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =======================
# Funções de Treinamento e Avaliação
# =======================
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    return total_loss / len(dataloader.dataset), accuracy.item()

# =======================
# Treinamento e Avaliação do Modelo Transformer
# =======================
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy = evaluate(model, test_loader, criterion)
    print(f"Época {epoch+1}/{num_epochs}")
    print(f"  Loss de Treino: {train_loss:.4f}")
    print(f"  Loss de Validação: {val_loss:.4f}")
    print(f"  Acurácia de Validação: {val_accuracy:.4f}")

# =======================
# Obter Previsões e Calcular Métricas
# =======================
def get_predictions(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

def calculate_metrics(labels, preds, model_name):
    report = classification_report(labels, preds, target_names=['true', 'fake'], zero_division=0, output_dict=True)
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    print(f"\nMétricas para o modelo {model_name}:")
    print(classification_report(labels, preds, target_names=['true', 'fake'], zero_division=0))
    return precision, recall, f1_score

def plot_metrics(metrics, model_name):
    labels = [label for label in metrics.keys() if isinstance(metrics[label], dict)]
    precision = [metrics[label].get('precision', 0) for label in labels]
    recall = [metrics[label].get('recall', 0) for label in labels]
    f1 = [metrics[label].get('f1-score', 0) for label in labels]

    x = range(len(labels))

    plt.figure(figsize=(10, 5))
    bars1 = plt.bar(x, precision, width=0.2, label='Precisão', color='b', align='center')
    bars2 = plt.bar([p + 0.2 for p in x], recall, width=0.2, label='Recall', color='g', align='center')
    bars3 = plt.bar([p + 0.4 for p in x], f1, width=0.2, label='F1-Score', color='r', align='center')

    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title(f'Métricas de Classificação para o Modelo {model_name}')
    plt.xticks([p + 0.2 for p in x], labels)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    plt.show()

# Previsões e Métricas
all_labels, all_preds = get_predictions(model, test_loader)
metrics = classification_report(all_labels, all_preds, target_names=['true', 'fake'], output_dict=True)
plot_metrics(metrics, "TransformerClassifier")

# =======================
# Matriz de Confusão
# =======================
def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['true', 'fake'], yticklabels=['true', 'fake'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Matriz de Confusão')
    plt.show()

plot_confusion_matrix(all_labels, all_preds)

# =======================
# Exemplo de Contagem de Palavras
# =======================
import numpy as np
df_combined['word_count'] = df_combined['message'].apply(lambda x: len(x.split()))
df_combined['log_word_count'] = np.log1p(df_combined['word_count'])

# =======================
# Classificação com RoBERTa
# =======================
# Definir parâmetros de treino
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
)

# Criar modelo RoBERTa
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Definir trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# =======================
# Classificação com Pipeline RoBERTa
# =======================
classifier_roberta = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

def classify_with_roberta(texts, classifier, max_length=10):
    preds = []
    for text in texts:
        truncated_text = text[:max_length]
        result = classifier(truncated_text)
        pred_label = result[0]['label']

        if pred_label in ['LABEL_2']:
            preds.append('true')
        else:
            preds.append('fake')
    return preds

# Classificar o conjunto de teste usando o RoBERTa
roberta_preds = classify_with_roberta(X_test.values, classifier_roberta)

# Comparar previsões com rótulos reais
y_test_str = [label_mapping[label] for label in y_test]

# Avaliar o modelo RoBERTa
print("Métricas para o modelo RoBERTa:")
metrics_roberta = classification_report(y_test_str, roberta_preds, target_names=['true', 'fake'], zero_division=0)
print(metrics_roberta)

# Exibir matriz de confusão para o RoBERTa
def plot_confusion_matrix_roberta(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=['true', 'fake'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['true', 'fake'], yticklabels=['true', 'fake'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Matriz de Confusão - RoBERTa')
    plt.show()

plot_confusion_matrix_roberta(y_test_str, roberta_preds)

# =======================
# Cálculo de Métricas para RoBERTa
# =======================
accuracy = accuracy_score(y_test, all_preds)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_str, roberta_preds, average='weighted')

# Exibir as métricas
print(f"Acurácia: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")
