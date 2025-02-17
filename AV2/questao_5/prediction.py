import torch
from AV2.questao_5.my_model import TextEncoder
import json
from AV2.utils.utils import tokenize_torch

# Vocab
vocab_path = "../models/modelo_do_0/vocab.json"
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)
VOCAB_SIZE = len(vocab) - 2

# modelo
model_path = "../models/modelo_do_0/text_encoder_my_model_weights.pt"
model = TextEncoder(vocab_size=VOCAB_SIZE, embedding_dim=256, hidden_dim=256, num_classes=2)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# inferência
texto_exemplo = "oi gordao arrombado"
input_tensor = tokenize_torch(texto_exemplo, vocab)

with torch.no_grad():
    output = model(input_tensor)

classe_predita = torch.argmax(output, dim=1).item()
print(f"Classe prevista: {classe_predita}")
