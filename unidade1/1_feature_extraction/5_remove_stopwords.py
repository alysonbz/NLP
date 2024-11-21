import spacy
from src.utils import load_gettyburg
gettysburg = load_gettyburg()

# Load model and create Doc object

nlp = spacy.load('en_core_web_sm')

stopwords = nlp.Defaults.stop_words

doc = nlp(gettysburg)

# Generate lemmatized tokens
lemmas = [token.lemma for token in doc]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas
            if lemma.isalpha() and lemma not in stopwords]

# Print string after text cleaning
print(' '.join(a_lemmas))