import spacy
from src.utils import load_gettyburg  # Importe sua função para carregar os dados

# Carregar o modelo en_core_web_sm do spaCy
nlp = spacy.load('en_core_web_sm')

# Carregar os dados usando sua função load_gettyburg (supondo que ela carrega um texto)
text = load_gettyburg()

# Criar um objeto Doc
doc = nlp(text)

# Gerar os tokens
tokens = [token.text for token in doc]
print(tokens)
