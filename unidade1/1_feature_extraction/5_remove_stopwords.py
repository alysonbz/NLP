import spacy
from src.utils import load_gettyburg

# Load the Gettysburg Address
gettysburg = load_gettyburg()

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(gettysburg)

# Generate lemmatized tokens
lemmas = [token.lemma_ for token in doc]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas
            if lemma.isalpha() and lemma not in nlp.Defaults.stop_words]

# Print string after text cleaning
print(' '.join(a_lemmas))
