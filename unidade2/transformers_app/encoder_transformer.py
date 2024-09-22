import torch
from unidade2.transformer_architeture.classifier_and_regression import ClassifierHead
from unidade2.transformer_architeture.transformer_encoder import TransformerEncoder

# Parâmetros do modelo
num_classes = 3
vocab_size = 15000      # Tamanho do vocabulário
batch_size = 16         # Tamanho do lote
model_dim = 512         # Dimensão do modelo
num_heads = 8           # Número de cabeças de atenção
num_layers = 6          # Número de camadas
feed_forward_dim = 1024 # Dimensão da rede feed-forward
sequence_length = 256   # Comprimento da sequência
dropout_rate = 0.2      # Taxa de dropout

# Criar sequência de entrada aleatória
input_sequence = torch.randint(0, vocab_size, (batch_size, sequence_length))
mask = (torch.rand(sequence_length, sequence_length) > 0.5).bool()  # Máscara de padding aleatória

# Instanciar o encoder e a cabeça de classificação
encoder = TransformerEncoder(vocab_size, model_dim, num_layers, num_heads, feed_forward_dim, dropout_rate, sequence_length)
classifier = ClassifierHead(model_dim, num_classes)

# Passar pela rede
encoder_output = encoder(input_sequence, mask)
classification_output = classifier(encoder_output)

print("Saídas de classificação para um lote de", batch_size, "sequências:")
print(classification_output)