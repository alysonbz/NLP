import spacy
from src.utils import load_gettyburg
gettysburg = load_gettyburg()

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web')

# Create a Doc object
doc = nlp(load_gettyburg())

# Generate tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]
print(pos)