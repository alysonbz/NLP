import spacy
from src.utils import load_gettyburg
gettysburg = load_gettyburg()

# Load model and create Doc object
nlp = ___

stopwords = ____

doc = __(gettysburg)

# Generate lemmatized tokens
lemmas = [___.___ for token in __]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in ___
            if lemma.___ and lemma not in ___]

# Print string after text cleaning
print(' '.join(___))