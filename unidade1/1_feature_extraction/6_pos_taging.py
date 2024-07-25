import spacy
from src.utils import load_gettyburg

# Carregar o texto de Gettysburg
gettysburg = load_gettyburg()

# Carregar o modelo en_core_web_sm do spaCy
nlp = spacy.load('en_core_web_sm')

# Criar um objeto Doc
doc = nlp(gettysburg)

# Gerar tokens e tags POS
pos = [(token.text, token.pos_) for token in doc]

# Imprimir tokens e tags POS
print(pos)
