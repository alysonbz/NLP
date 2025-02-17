import pandas as pd
import torch
import torch.nn as nn


def tokenize(text):
    text = text.lower().replace(",", "").replace(".", "")
    tokens = text.split()  # Separar por espaço
    return tokens


def text_to_indices(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenize(text)]


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        return self.encoder(x, mask)


if __name__ == "__main__":
    df = pd.read_csv("../subs/df_train_balanced.csv")
    texts = df['text'].dropna().tolist()
    all_tokens = set(token for text in texts for token in tokenize(text))
    vocab = {word: idx for idx, word in enumerate(all_tokens, start=1)}

    # Adicionando tokens especiais
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = len(vocab) + 1  # Token para palavras desconhecidas

    # Parâmetros do modelo
    d_model = 8  # Dimensão do embedding
    num_heads = 2  # Número de cabeças de atenção
    d_ff = 16  # Dimensão da Feed-Forward Network
    num_layers = 2  # Número de camadas do Encoder
    vocab_size = len(vocab)

    encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, vocab_size)

    for text in texts[1:10]:
        input_indices = torch.tensor(text_to_indices(text, vocab)).unsqueeze(0)

        # Passa pelo encoder
        output = encoder(input_indices)

        # Impressão dos resultados
        print("\nTexto de Entrada:", text)
        print("Tokens:", input_indices)
        print("\nSaída do Encoder (Shape):", output.shape)
        print("Saída do Encoder (Valores):\n", output)
