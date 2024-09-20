import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def slit_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3).view(batch_size * self.n_heads, -1, self.d_k)
        
    def compute_attention(self, query, key, value, mask=None):
        scores = torch.matmul(q)
  
        
        return out