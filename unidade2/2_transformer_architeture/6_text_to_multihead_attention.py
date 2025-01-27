import numpy as np
from gensim.models import Word2Vec

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len

    def get_positional_encoding(self, seq_len):
        """
        Calcula a codificação posicional.
        :param seq_len: Comprimento da sequência.
        :return: Matriz de codificação posicional.
        """
        PE = np.zeros((seq_len, self.d_model))
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                # **Complete o cálculo do seno e cosseno**
                PE[pos, i] = None  # Seno para posições pares
                if i + 1 < self.d_model:
                    PE[pos, i + 1] = None  # Cosseno para posições ímpares
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

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Calcula a atenção escalonada por produto escalar.
        :param Q: Matriz de consultas.
        :param K: Matriz de chaves.
        :param V: Matriz de valores.
        :return: Saída e pesos de atenção.
        """
        # **Complete o cálculo**
        matmul_qk = None  # Produto escalar entre Q e K^T
        scaled_attention_logits = None  # Escalonar pelo tamanho de d_k
        attention_weights = None  # Aplicar softmax para obter os pesos de atenção
        output = None  # Multiplicar os pesos pela matriz V

        return output, attention_weights

    def split_heads(self, X):
        """
        Divide as matrizes Q, K, e V em múltiplas cabeças.
        :param X: Matriz a ser dividida (Q, K ou V).
        :return: Matriz reformatada para múltiplas cabeças.
        """
        # **Complete a divisão**
        batch_size, seq_len, d_model = X.shape
        X = None  # Redimensionar para (batch_size, seq_len, num_heads, d_k)
        return None  # Reordenar os eixos para (batch_size, num_heads, seq_len, d_k)

    def forward(self, Q, K, V):
        """
        Executa o processo de Multi-Head Attention.
        """
        # Projeções lineares
        Q_proj = None  # Projeção para Q
        K_proj = None  # Projeção para K
        V_proj = None  # Projeção para V

        # Divisão em múltiplas cabeças
        Q_heads = None
        K_heads = None
        V_heads = None

        # Cálculo da atenção
        head_outputs = []
        for i in range(self.num_heads):
            Q_i = Q_heads[:, i, :, :]
            K_i = K_heads[:, i, :, :]
            V_i = V_heads[:, i, :, :]
            output, _ = self.scaled_dot_product_attention(Q_i, K_i, V_i)
            head_outputs.append(output)

        # Concatenar as saídas das cabeças
        concatenated = None

        # Projeção final
        output = None

        return output


if __name__ == "__main__":
    # Texto de entrada
    corpus = [
        "O aprendizado profundo é fascinante".split(),
        "Modelos de linguagem transformaram o NLP".split(),
        "Multi-Head Attention é um conceito poderoso".split(),
        "Codificação posicional ajuda no aprendizado de sequência".split()
    ]

    embedding_dim = 8
    w2v_model = Word2Vec(sentences=corpus, vector_size=embedding_dim, window=5, min_count=1, workers=4)

    text = "O aprendizado profundo é fascinante"
    tokens = text.split()

    embeddings = np.array([w2v_model.wv[token] for token in tokens])
    embeddings = embeddings[np.newaxis, :, :]  # (batch_size, seq_len, embedding_dim)

    pos_enc = PositionalEncoding(embedding_dim)
    positional_encoding = pos_enc.get_positional_encoding(len(tokens))
    embeddings += positional_encoding[np.newaxis, :, :]  # Adiciona codificação posicional

    num_heads = 2
    mha = MultiHeadAttention(embedding_dim, num_heads)

    Q = embeddings
    K = embeddings
    V = embeddings

    output = mha.forward(Q, K, V)
    print("Saída do Multi-Head Attention:\n", output)
