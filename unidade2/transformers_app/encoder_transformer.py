import torch
from transformers.tools.evaluate_agent import classifier
###
from unidade2.transformer_architeture.classifier_and_regression import ClassifierHead
from unidade2.transformer_architeture.transformer_encoder import TransformerEncoder
from unidade2.transformer_architeture.encoder_layer import EncoderLayer

# Definições de parâmetros
num_classes = 3
vocab_size = 10000
batch_size = 8
d_model = 512
num_heads = 16
num_layers = 4
d_ff = 2048
sequence_length = 32
dropout = 0.1

# Entrada do encoder
input_sequence = torch.randint(0, vocab_size, (batch_size, sequence_length))
mask = torch.randint(0, 2, (sequence_length, sequence_length))

# Instanciar o corpo e o cabeçalho do transformer encoder
encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_sequence_length=sequence_length, dropout=0.1)
classifier = ClassifierHead(d_model, num_classes)

# Completar a passagem forward
output = encoder(input_sequence, mask)
classification = classifier(output)

print("Classification outputs for a batch of", batch_size, "sequences:")
print(classification)