import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.1):
        super(ClassifierHead, self).__init__()

        # Camada linear para mapear a saída do encoder para o número de classes
        self.fc = nn.Linear(d_model, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor de entrada de forma (batch_size, seq_len, d_model).
        """
        # Aplicar dropout
        x = self.dropout(x)

        # Aplicar a camada linear para obter logits para cada classe
        logits = self.fc(x)

        return logits


class RegressionHead(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(RegressionHead, self).__init__()

        # Camada linear para mapear a saída do encoder para um valor contínuo
        self.fc = nn.Linear(d_model, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor de entrada de forma (batch_size, seq_len, d_model).
        """
        # Aplicar dropout
        x = self.dropout(x)

        # Aplicar a camada linear para obter a previsão contínua
        predictions = self.fc(x)

        return predictions
# Exemplo de inicialização para classificação
classifier = ClassifierHead(d_model=512, num_classes=10, dropout=0.1)

# Inputs de exemplo
src = torch.rand(32, 20, 512)  # (batch_size, seq_len, d_model)

# Cálculo dos logits de classificação
logits = classifier(src)
print(logits.shape)  # Deve ser (32, 20, 10)

# Exemplo de inicialização para regressão
regressor = RegressionHead(d_model=512, dropout=0.1)

# Cálculo das previsões de regressão
predictions = regressor(src)
print(predictions.shape)  # Deve ser (32, 20, 1)
