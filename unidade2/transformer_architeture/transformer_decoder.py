import torch.nn as nn
import torch.nn.functional as F
from unidade2.transformer_architeture.positional_encoding import PositionalEncoder
from unidade2.transformer_architeture.decoder_layer import DecoderLayer

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads,d_ff, max_seq_len, dropout):
        super(TransformerDecoder, self).__init__()
        self.embeding = nn.Embedding(vocab_size,d_model)
        self.positional_encoding = PositionalEncoder(d_model, max_seq_len)
        self.layer = nn.ModuleList([DecoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model,vocab_size)


    def forward(self,x, self_mask,encoder_output,cross_mask):
        x = self.embeding(x)
        x = self.positional_encoding(x)
        for layer in self.layer:
            x = layer(x, self_mask,encoder_output,cross_mask)

        x= self.fc(x)
        return F.log_softmax(x,dim = -1)