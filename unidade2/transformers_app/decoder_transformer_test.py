import torch
from unidade2.transformer_architeture.transformer_decoder import TransformerDecoder


def generate_random_cross_attention_mask(batch_size, sequence_length):
    # Gera uma máscara aleatória com 0s e 1s
    mask = torch.randint(0, 2, (batch_size, 1, sequence_length)).bool()
    return mask

# Definições de parâmetros
vocab_size = 15000      # Tamanho do vocabulário
batch_size = 16         # Tamanho do lote
model_dim = 512         # Dimensão do modelo
num_heads = 8           # Número de cabeças de atenção
num_layers = 6          # Número de camadas
feed_forward_dim = 1024 # Dimensão da rede feed-forward
sequence_length = 256   # Comprimento da sequência
dropout_rate = 0.2      # Taxa de dropout

# Entrada do decoder (tgt)
input_sequence = torch.randint(0, vocab_size, (batch_size, sequence_length))

# Saída do encoder simulada
encoder_output = torch.rand(batch_size, sequence_length, model_dim)

# Máscara de self-attention (máscara causal para impedir vazamento de informações futuras)
self_attention_mask = (1 - torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)).bool()

# Máscara de cross-attention
cross_attention_mask = generate_random_cross_attention_mask(batch_size, sequence_length)

# Instanciar o decoder
decoder = TransformerDecoder(vocab_size, model_dim, num_layers, num_heads, feed_forward_dim, dropout_rate, sequence_length)

# Passar pelo decoder
output = decoder(input_sequence, self_attention_mask, encoder_output, cross_attention_mask)

# Verificar a forma da saída
print(output.shape)
