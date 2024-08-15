import spacy
from src.utils import load_gettyburg

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')

getty = load_gettyburg()

# Create a Doc object
doc = nlp(getty)

# Generate the tokens
tokens = [token.text for token in doc]
print(tokens)