import spacy
from src.utils import load_gettyburg
gettysburg = load_gettyburg()

# Load the en_core_web_sm model
nlp = spacy.load(____)

# Create a Doc object
doc = ___(___)

# Generate tokens and pos tags
pos = [(token.____, token.____) for token in doc]
print(pos)