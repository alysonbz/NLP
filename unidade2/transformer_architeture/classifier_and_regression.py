import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc_layer = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return F.log_softmax(self.fc_layer(x), dim=-1)

class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc_layer(x)
