import torch.nn as nn
from unidade2.transformer_architeture.attention_mecanism import MultiHeadAttention
from unidade2.transformer_architeture.feedFowardSubLayer import FeedForwardSubLayer

class DecoderLayer(nn.Module):
    def __init__(self,d_model, num_heads, d_ff, dropout):
        super(DecoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self ,x,causal_mask, encoder_output, cross_mask):
        self_attn_output = self.self_attn(x ,x ,x, causal_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attn(x, encoder_output,encoder_output,cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x +  self.dropout(ff_output))
        return ximport torch.nn as nn
from unidade2.transformer_architeture.attention_mecanism import MultiHeadAttention
from unidade2.transformer_architeture.feedFowardSubLayer import FeedForwardSubLayer

class DecoderLayer(nn.Module):
    def __init__(self,d_model, num_heads, d_ff, dropout):
        super(DecoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self ,x,causal_mask, encoder_output, cross_mask):
        self_attn_output = self.self_attn(x ,x ,x, causal_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attn(x, encoder_output,encoder_output,cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x +  self.dropout(ff_output))
        return x