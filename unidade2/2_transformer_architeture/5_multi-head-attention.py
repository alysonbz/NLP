import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads."
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_q = np.random.rand(d_model, d_model)
        self.W_k = np.random.rand(d_model, d_model)
        self.W_v = np.random.rand(d_model, d_model)

        self.b_q = np.random.rand(d_model)
        self.b_k = np.random.rand(d_model)
        self.b_v = np.random.rand(d_model)

        self.W_o = np.random.rand(d_model, d_model)
        self.b_o = np.random.rand(d_model)

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def scaled_dot_product_attention(self, Q, K, V):
        # Q, K, V: (batch_size, seq_len, d_k)
        matmul_qk = np.matmul(Q, K.transpose(0, 2, 1))
        scaled_attention_logits = matmul_qk / np.sqrt(self.d_k)
        attention_weights = self.softmax(scaled_attention_logits)
        output = np.matmul(attention_weights, V)
        return output, attention_weights

    def split_heads(self, X):
        # X: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = X.shape
        X = X.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return X.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_k)

    def forward(self, Q, K, V):
        # Passo 1: Projeções lineares
        Q_proj = Q @ self.W_q + self.b_q
        K_proj = K @ self.W_k + self.b_k
        V_proj = V @ self.W_v + self.b_v

        # Passo 2: Dividir em múltiplas cabeças
        Q_heads = self.split_heads(Q_proj)
        K_heads = self.split_heads(K_proj)
        V_heads = self.split_heads(V_proj)

        # Passo 3: Atenção por cabeça
        head_outputs = []
        for i in range(self.num_heads):
            Q_i = Q_heads[:, i, :, :]
            K_i = K_heads[:, i, :, :]
            V_i = V_heads[:, i, :, :]
            output, _ = self.scaled_dot_product_attention(Q_i, K_i, V_i)
            head_outputs.append(output)

        # Passo 4: Concatenar cabeças
        concatenated = np.concatenate(head_outputs, axis=-1)

        # Passo 5: Projeção final
        output = concatenated @ self.W_o + self.b_o

        return output


# Exemplo de uso
if __name__ == "__main__":
    np.random.seed(42)

    d_model = 6
    num_heads = 3
    seq_len = 4
    batch_size = 1

    Q = np.random.rand(batch_size, seq_len, d_model)
    K = np.random.rand(batch_size, seq_len, d_model)
    V = np.random.rand(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model, num_heads)
    output = mha.forward(Q, K, V)

    print("Saída do Multi-Head Attention:\n", output)