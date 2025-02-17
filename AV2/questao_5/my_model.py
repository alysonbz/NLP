import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super(TextEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim ** -0.5)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, lengths=None):
        x = self.embedding(input_ids)

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        _, (hidden, _) = self.lstm(x)

        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        x = self.fc1(hidden)
        x = self.batch_norm(x)
        x = torch.relu(x)  # Pode testar com GELU
        x = self.dropout(x)
        x = self.fc2(x)

        return x
