#from 3_attention_mechanism import MultiHeadAttention
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_leads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x+self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x+self.dropout(attn_output))
        return x