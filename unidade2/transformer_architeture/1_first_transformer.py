import torch.nn as nn


model = nn.Transformer(
    d_model=512,
    nhead=2,
    num_encoder_layers=3,
    num_decoder_layers=3
)

print(model)