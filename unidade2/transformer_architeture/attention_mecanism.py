import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        # Verificação para garantir que d_model seja divisível pelo número de cabeças
        assert d_model % n_heads == 0, "d_model deve ser divisível por n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads  # Dimensão de cada cabeça de atenção

        # Camadas lineares para projetar as queries, keys, e values
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # Camada linear para projetar a saída após a atenção
        self.fc_out = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Divide os embeddings em múltiplas cabeças de atenção.
        A entrada x tem o formato (batch_size, seq_len, d_model).
        Após a divisão, o formato será (batch_size, n_heads, seq_len, depth).
        """
        x = x.view(batch_size, -1, self.n_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, depth)

    def compute_attention(self, query, key, value, mask=None):
        """
        Computa a atenção usando as queries, keys e values.
        Retorna o valor ponderado pela atenção e os pesos de atenção.
        """
        # Produto escalar entre query e key, dividido pela raiz quadrada da profundidade
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))

        # Aplicação da máscara, se fornecida
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax para normalizar os scores em probabilidades
        attention_weights = F.softmax(scores, dim=-1)

        # Produto entre os pesos de atenção e os valores (values)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Projeção das queries, keys, e values
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Dividir em múltiplas cabeças de atenção
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Calcular a atenção
        attention_output, attention_weights = self.compute_attention(query, key, value, mask)

        # Concatenar as cabeças de atenção (voltar ao formato original)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)

        # Aplicar a camada linear final
        output = self.fc_out(attention_output)

        return output, attention_weights
