import torch
import torch.nn as nn

class FeedForwardSubLayer(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super(FeedForwardSubLayer, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

