import torch
from unidade2.transformer_architeture.transformer_encoder import TransformerEncoder
from unidade2.transformer_architeture.transformer_decoder import TransformerDecoder


# Definições de parâmetros
vocab_size = 15000      # Tamanho do vocabulário
batch_size = 16         # Tamanho do lote
model_dim = 512         # Dimensão do modelo
num_heads = 8           # Número de cabeças de atenção
num_layers = 6          # Número de camadas
feed_forward_dim = 1024 # Dimensão da rede feed-forward
sequence_length = 256   # Comprimento da sequência
dropout_rate = 0.2      # Taxa de dropout

# Criar um lote de sequências de entrada aleatórias
input_sequence = torch.randint(0, vocab_size, (batch_size, sequence_length))
padding_mask = (torch.rand(sequence_length, sequence_length) > 0.5).bool()  # Máscara de padding aleatória
causal_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()  # Máscara causal

# Instanciar os componentes do Transformer
encoder = TransformerEncoder(vocab_size, model_dim, num_layers, num_heads, feed_forward_dim, dropout_rate, sequence_length)
decoder = TransformerDecoder(vocab_size, model_dim, num_layers, num_heads, feed_forward_dim, dropout_rate, sequence_length)

# Passar as máscaras necessárias como argumentos para o encoder e o decoder
encoder_output = encoder(input_sequence, padding_mask)
decoder_output = decoder(input_sequence, causal_mask, encoder_output, padding_mask)

print("Forma da saída do lote: ", decoder_output.shape)