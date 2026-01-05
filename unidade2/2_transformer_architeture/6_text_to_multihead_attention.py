import numpy as np
from gensim.models import Word2Vec


class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len

    def get_positional_encoding(self, seq_len):
        PE = np.zeros((seq_len, self.d_model))
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                PE[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                if i + 1 < self.d_model:
                    PE[pos, i + 1] = np.cos(pos / (10000 ** (i / self.d_model)))
        return PE


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
        matmul_qk = np.matmul(Q, K.transpose(0, 2, 1))
        scaled_attention_logits = matmul_qk / np.sqrt(self.d_k)
        attention_weights = self.softmax(scaled_attention_logits)
        output = np.matmul(attention_weights, V)
        return output, attention_weights

    def split_heads(self, X):
        batch_size, seq_len, _ = X.shape
        X = X.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return X.transpose(0, 2, 1, 3)

    def forward(self, Q, K, V):
        # Projeções lineares
        Q_proj = Q @ self.W_q + self.b_q
        K_proj = K @ self.W_k + self.b_k
        V_proj = V @ self.W_v + self.b_v

        # Divisão em múltiplas cabeças
        Q_heads = self.split_heads(Q_proj)
        K_heads = self.split_heads(K_proj)
        V_heads = self.split_heads(V_proj)

        # Atenção por cabeça
        head_outputs = []
        for i in range(self.num_heads):
            Q_i = Q_heads[:, i, :, :]
            K_i = K_heads[:, i, :, :]
            V_i = V_heads[:, i, :, :]
            output, _ = self.scaled_dot_product_attention(Q_i, K_i, V_i)
            head_outputs.append(output)

        # Concatenação das cabeças
        concatenated = np.concatenate(head_outputs, axis=-1)

        # Projeção final
        output = concatenated @ self.W_o + self.b_o

        return output


if __name__ == "__main__":
    np.random.seed(42)

    # Corpus de exemplo
    corpus = [
        "O aprendizado profundo é fascinante".split(),
        "Modelos de linguagem transformaram o NLP".split(),
        "Multi-Head Attention é um conceito poderoso".split(),
        "Codificação posicional ajuda no aprendizado de sequência".split()
    ]

    embedding_dim = 8
    w2v_model = Word2Vec(
        sentences=corpus,
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        workers=4
    )

    text = "O aprendizado profundo é fascinante"
    tokens = text.split()

    embeddings = np.array([w2v_model.wv[token] for token in tokens])
    embeddings = embeddings[np.newaxis, :, :]  # (batch_size, seq_len, d_model)

    pos_enc = PositionalEncoding(embedding_dim)
    positional_encoding = pos_enc.get_positional_encoding(len(tokens))
    embeddings += positional_encoding[np.newaxis, :, :]

    num_heads = 2
    mha = MultiHeadAttention(embedding_dim, num_heads)

    Q = embeddings
    K = embeddings
    V = embeddings

    output = mha.forward(Q, K, V)
    print("Saída do Multi-Head Attention:\n", output)