import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        Inicializa o bloco de Multi-Head Attention.
        :param d_model: Dimensão do embedding (modelo).
        :param num_heads: Número de cabeças de atenção.
        """
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads."
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # Dimensão de cada cabeça

        # Matrizes de projeção para Q, K, V
        self.W_q = np.random.rand(d_model, d_model)  # Projeção para Q
        self.W_k = np.random.rand(d_model, d_model)  # Projeção para K
        self.W_v = np.random.rand(d_model, d_model)  # Projeção para V

        # Biases para Q, K, V
        self.b_q = np.random.rand(d_model)
        self.b_k = np.random.rand(d_model)
        self.b_v = np.random.rand(d_model)

        # Matriz de projeção final (após concatenação)
        self.W_o = np.random.rand(d_model, d_model)  # Projeção de saída
        self.b_o = np.random.rand(d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Calcula a atenção escalonada por produto escalar.
        :param Q: Matriz de consultas.
        :param K: Matriz de chaves.
        :param V: Matriz de valores.
        :return: A saída da atenção e os pesos de atenção.
        """
        matmul_qk = np.matmul(Q, K.swapaxes(-2, -1))  # Produto escalar entre Q e K^T
        scaled_attention_logits = matmul_qk / np.sqrt(self.d_k)  # Escalonamento
        attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)  # Softmax
        output = np.matmul(attention_weights, V)  # Aplicar pesos a V
        return output, attention_weights

    def split_heads(self, X):
        """
        Divide as matrizes Q, K e V em múltiplas cabeças.
        :param X: Matriz a ser dividida (Q, K ou V).
        :return: Matriz reformatada para múltiplas cabeças.
        """
        batch_size, seq_len, d_model = X.shape
        X = X.reshape(batch_size, seq_len, self.num_heads, self.d_k)  # Redimensionar
        return X.transpose(0, 2, 1, 3)  # Reorganizar eixos para (batch_size, num_heads, seq_len, d_k)

    def forward(self, Q, K, V):
        """
        Executa o processo de Multi-Head Attention.
        :param Q: Matriz de consultas.
        :param K: Matriz de chaves.
        :param V: Matriz de valores.
        :return: Saída do bloco de Multi-Head Attention.
        """
        Q_proj = np.matmul(Q, self.W_q) + self.b_q  # Projeção de Q
        K_proj = np.matmul(K, self.W_k) + self.b_k  # Projeção de K
        V_proj = np.matmul(V, self.W_v) + self.b_v  # Projeção de V

        # Dividir em múltiplas cabeças
        Q_heads = self.split_heads(Q_proj)
        K_heads = self.split_heads(K_proj)
        V_heads = self.split_heads(V_proj)

        # Aplicar atenção em cada cabeça
        head_outputs = []
        for i in range(self.num_heads):
            Q_i = Q_heads[:, i, :, :]
            K_i = K_heads[:, i, :, :]
            V_i = V_heads[:, i, :, :]
            output, _ = self.scaled_dot_product_attention(Q_i, K_i, V_i)
            head_outputs.append(output)

        # Concatenar as saídas de todas as cabeças
        concatenated = np.concatenate(head_outputs, axis=-1)

        # Aplicar a camada linear final
        output = np.matmul(concatenated, self.W_o) + self.b_o
        return output

# Exemplo de uso
if __name__ == "__main__":
    np.random.seed(42)

    # Parâmetros do modelo
    d_model = 6  # Dimensão do modelo (embedding)
    num_heads = 3  # Número de cabeças de atenção
    seq_len = 4  # Comprimento da sequência
    batch_size = 1  # Tamanho do batch

    # Matrizes de entrada (Q, K, V)
    Q = np.random.rand(batch_size, seq_len, d_model)  # Exemplo de consultas
    K = np.random.rand(batch_size, seq_len, d_model)  # Exemplo de chaves
    V = np.random.rand(batch_size, seq_len, d_model)  # Exemplo de valores

    # Criar o bloco de Multi-Head Attention
    mha = MultiHeadAttention(d_model, num_heads)

    # Executar o forward pass
    output = mha.forward(Q, K, V)
    print("Saída do Multi-Head Attention:\n", output)
