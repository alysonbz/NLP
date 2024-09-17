import torch
from unidade2.transformer_architeture.transformer_decoder import TransformerDecoder


# Definições de parâmetros
vocab_size = 10000
batch_size = 8
d_model = 256
num_heads = 16
num_layers = 8
d_ff = 512
sequence_length = 128
dropout = 0.1

# Entrada do decoder (tgt)
input_sequence = torch.randint(0, vocab_size, (batch_size, sequence_length))

# Saída do encoder simulada
encoder_output = torch.rand(batch_size, sequence_length, d_model)

# Máscara de self-attention (máscara causal para impedir vazamento de informações futuras)
self_attention_mask = (1 - torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)).bool()

# Máscara de cross-attention (pode ser None se você não tiver tokens de padding no encoder_output)
cross_attention_mask = None  # Defina conforme necessário

# Instanciar o decoder
decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, sequence_length)

# Passar pelo decoder
output = decoder(input_sequence, self_attention_mask, encoder_output, cross_attention_mask)

# Verificar a forma da saída
print(output.shape)