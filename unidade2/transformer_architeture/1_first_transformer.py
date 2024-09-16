import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, d_model=512, n_heads=2, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1):
        super(TransformerModel, self).__init__()

        # Transformer parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # Encoder and Decoder layers with batch_first=True
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                                                        dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                                                        dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Embedding for source and target sequences
        self.embedding = nn.Embedding(1000, d_model)  # Assumes vocab size of 1000
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 500, d_model))  # Positional encoding (assuming max seq length 500)

        # Output layer
        self.fc_out = nn.Linear(d_model, 1000)  # Output layer (same vocab size)

    def forward(self, src, tgt):
        # Embedding and positional encoding
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        # Encoder
        memory = self.encoder(src)

        # Decoder
        output = self.decoder(tgt, memory)

        # Output layer
        output = self.fc_out(output)

        return output


# Example usage
model = TransformerModel()
src = torch.randint(0, 1000, (32, 10))  # (batch size, sequence length)
tgt = torch.randint(0, 1000, (32, 20))

output = model(src, tgt)
print(output.shape)  # Output shape will be (batch_size, tgt_seq_len, vocab_size)
