import spacy
from src.utils import load_gettyburg

# Carregar o modelo en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Carregar o discurso de Gettysburg
text = load_gettyburg()

# Criar um objeto Doc
doc = nlp(text)

# Gerar os tokens
tokens = [token.text for token in doc]
print(tokens)
