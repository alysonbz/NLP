import torch
import torch.nn as nn

# hiperparâmetros para arquitetura
d_model = 512
n_heads = 2
num_encoder_layers = 6
num_decoder_layers = 6

# arquitetura Transformer
model = nn.Transformer(
    d_model=d_model,
    nhead=n_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers
)

# Exibindo a arquitetura do modelo
print(model)