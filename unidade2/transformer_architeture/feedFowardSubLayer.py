import torch.nn as nn

class FeedForwardSubLayer(nn.Module):
    def __init__(self,d_model, d_ff):
        super(FeedForwardSubLayer, self).__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.relu = nn.Relu()

    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))