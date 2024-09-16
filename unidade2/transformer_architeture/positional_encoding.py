import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=500, dropout=0.1):
        super(PositionalEncoder, self).__init__()

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of positional encodings (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Sinusoids for even indices, cosines for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        # Add a batch dimension to pe
        pe = pe.unsqueeze(0)  # Shape becomes (1, max_seq_len, d_model)

        # Register pe as a buffer so it is saved and moved to device with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encodings to the input embeddings
        x = x + self.pe[:, :x.size(1), :]

        # Apply dropout
        return self.dropout(x)
