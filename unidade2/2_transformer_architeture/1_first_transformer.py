import torch


transformer = torch.nn.Transformer(
    d_model=512,
    nhead=2,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)

print(transformer)
