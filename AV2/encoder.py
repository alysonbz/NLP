import torch
import torch.nn as nn
import pandas as pd

# ---------------------- 1. CARREGAR O DATASET ----------------------
df = pd.read_csv("../AV2/dataset/twitter_sentiment.csv")

texts = df["tweet_text"].astype(str).tolist()


# ---------------------- 2. TOKENIZAÇÃO E CRIAÇÃO DO VOCABULÁRIO ----------------------
def tokenize(text):
    """Tokeniza o texto removendo pontuações e transformando em minúsculas."""
    text = text.lower().replace(",", "").replace(".", "")
    tokens = text.split()
    return tokens


# Criar vocabulário a partir do dataset
all_tokens = set(token for sentence in texts for token in tokenize(sentence))
vocab = {word: idx + 1 for idx, word in enumerate(all_tokens)}
vocab["<PAD>"] = 0  # Padding
vocab["<UNK>"] = len(vocab)  # Token desconhecido


def text_to_indices(text, vocab):
    """Converte um texto para uma sequência de índices."""
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]


# ---------------------- 3. IMPLEMENTAR O ENCODER ----------------------
class ManualEncoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(ManualEncoder, self).__init__()

        # Camada de embedding para representar tokens como vetores densos
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Rede neural para refinar a representação dos embeddings
        self.encoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.embedding(x)  # Converter índices em vetores densos
        x = self.encoder(x)  # Refinar representações com rede neural
        return x


# ---------------------- 4. TESTANDO O ENCODER ----------------------
# Definir hiperparâmetros
d_model = 50  # Dimensão dos embeddings
vocab_size = len(vocab) + 1  # Tamanho do vocabulário

# Criar modelo
model = ManualEncoder(vocab_size, d_model)

# Converter alguns exemplos do dataset em índices
sample_sentences = texts[:5]
indices_list = [text_to_indices(sentence, vocab) for sentence in sample_sentences]

# Padding manual para manter o mesmo comprimento
max_len = max(len(seq) for seq in indices_list)
indices_list = [seq + [0] * (max_len - len(seq)) for seq in indices_list]  # Padding

# Converter para tensor
indices_tensor = torch.tensor(indices_list)

# Passar pelo encoder
encoded_output = model(indices_tensor)
print("Saída do encoder (shape):", encoded_output.shape)
print(encoded_output[0])  # Vetores de cada palavra do primeiro tweet

