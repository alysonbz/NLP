import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, model_dim, feed_forward_dim):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, feed_forward_dim)
        self.fc2 = nn.Linear(feed_forward_dim, model_dim)
        self.activation = nn.ReLU()

    def _apply_feed_forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

    def forward(self, x):
        return self._apply_feed_forward(x)
