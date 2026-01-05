import numpy as np

def softmax(x):
    # Passo 1: Subtrair o valor máximo de x para estabilidade numérica
    # **Complete este passo**
    x = x - np.max(x, axis=-1, keepdims=True)

    # Passo 2: Calcular o exponencial de cada elemento de x
    # **Complete este passo**
    x = np.exp(x)

    # Passo 3: Normalizar os valores exponenciais, dividindo cada valor pelo somatório dos exponenciais
    # **Complete este passo**
    x = x / np.sum(x, axis=-1, keepdims=True)

    return x  # Retorne a versão normalizada de x

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
        # Passo 1: Calcular o produto escalar de Q e K^T
        # **Complete o cálculo aqui**
        scores = np.matmul(Q, K.transpose(0, 2, 1))

        # Passo 2: Escalonar os resultados dividindo por sqrt(d_k)
        scaled_attention_logits = scores / np.sqrt(self.d_k)  # **Complete esta parte**

        # Passo 3: Aplicar softmax para obter as probabilidades
        attention_weights = softmax(scaled_attention_logits)

        # Passo 4: Multiplicar as probabilidades pela matriz de valores V
        output = np.dot(attention_weights, V)

        return output, attention_weights

    def split_heads(self, X):
        """
        Divide a matriz em múltiplas cabeças.
        :param X: Matriz original (batch_size, seq_len, d_model).
        :param num_heads: Número de cabeças.
        :return: Matriz reformatada (batch_size, num_heads, seq_len, d_k).
        """
        # Complete as operações abaixo para dividir X em múltiplas cabeças.
        batch_size, seq_len, d_model = X.shape
        d_k = d_model // self.num_heads  # Dimensão de cada cabeça
        # Redimensione X para incluir o número de cabeças e a nova dimensão d_k
        # **Complete aqui**
        X = X.reshape(batch_size, seq_len, self.num_heads, d_k)

        # Reordene os eixos para (batch_size, num_heads, seq_len, d_k)
        # **Complete aqui**

        X = np.transpose(X, (0, 2, 1, 3))

        return X  # Retorne a matriz transposta dividida corretamente.

    def forward(self, Q, K, V):
        """
        Executa o processo de Multi-Head Attention.
        :param Q: Matriz de consultas.
        :param K: Matriz de chaves.
        :param V: Matriz de valores.
        :return: Saída do bloco de Multi-Head Attention.
        """
        # Passo 1: Aplicar as camadas lineares para projetar Q, K, V
        Q_proj = Q @ self.W_q + self.b_q  # (batch_size, seq_len, d_model)
        K_proj = K @ self.W_k + self.b_k  # (batch_size, seq_len, d_model)
        V_proj = V @ self.W_v + self.b_v  # (batch_size, seq_len, d_model)

        # Passo 2: Dividir as matrizes projetadas em múltiplas cabeças
        # **Use a função split_heads para dividir Q_proj, K_proj e V_proj**
        Q_heads = self.split_heads(Q_proj)  # (batch_size, num_heads, seq_len, d_k)
        K_heads = self.split_heads(K_proj)  # (batch_size, num_heads, seq_len, d_k)
        V_heads = self.split_heads(V_proj)  # (batch_size, num_heads, seq_len, d_k)

        # Passo 3: Aplicar atenção em cada cabeça
        head_outputs = []
        for i in range(self.num_heads):
            Q_i = Q_heads[:, i, :, :]  # Seleciona a cabeça i
            K_i = K_heads[:, i, :, :]
            V_i = V_heads[:, i, :, :]
            output, _ = self.scaled_dot_product_attention(Q_i, K_i, V_i)
            head_outputs.append(output)

        # Passo 4: Concatenar as saídas de todas as cabeças
        concatenated = np.concatenate(head_outputs, axis=-1)  # Concatenar as saídas das cabeças

        # Passo 5: Aplicar a camada linear final
        output = concatenated @ self.W_o + self.b_o  # Projeção final após concatenar as cabeças

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
