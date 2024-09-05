import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, dimensao_entrada, numero_classes):
        super(ClassifierHead, self).__init__()

        self.camada_linear = nn.Linear(dimensao_entrada, numero_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tensor_entrada):
        logits = self.camada_linear(tensor_entrada)
        saida_classificacao = self.softmax(logits)
        return saida_classificacao


class RegressionHead(nn.Module):
    def __init__(self, dimensao_entrada, dimensao_saida):
        super(RegressionHead, self).__init__()

        self.camada_linear = nn.Linear(dimensao_entrada, dimensao_saida)

    def forward(self, tensor_entrada):
        saida_regressao = self.camada_linear(tensor_entrada)
        return saida_regressao
