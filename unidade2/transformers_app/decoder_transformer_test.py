import torch
from unidade2.transformer_architeture.transformer_decoder import TransformerDecoder  # Ajuste o caminho de importação

# Definindo parâmetros para o teste
d_model = 512
n_heads = 8
num_layers = 6
dim_feedforward = 2048
batch_size = 64
sequence_length = 10
target_sequence_length = 12

# Criando uma instância do TransformerDecoder
decoder = TransformerDecoder(d_model, n_heads, num_layers, dim_feedforward)

# Criando entradas aleatórias para o decodificador
encoder_output = torch.randn(batch_size, sequence_length, d_model)  # Saída do encoder (memória)
target_input = torch.randn(batch_size, target_sequence_length, d_model)  # Entrada do decodificador (target)

# Máscaras opcionais para evitar que o decodificador veja futuras posições
target_mask = torch.triu(torch.ones(target_sequence_length, target_sequence_length), diagonal=1).bool()
memory_mask = None  # Máscara para a memória (saída do encoder), pode ser None

# Executando o decodificador
output = decoder(target_input, encoder_output, tgt_mask=target_mask, memory_mask=memory_mask)

# Exibindo as dimensões da saída
print("Dimensão da saída:", output.size())
