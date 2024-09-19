#classifier_and_regression.py
import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.fc(x)


class RegressionHead(nn.Module):
    def __init__(self, d_model):
        super(RegressionHead, self).__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.fc(x)
