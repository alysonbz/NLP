import torch.nn as nn
import torch.nn.functional as F

class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)

class RegressionHead(nn.Module):
    def __init__(self, d_model,output_dim):
        super(RegressionHead,self).__init__()
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self,x):
        return self.fc(x)