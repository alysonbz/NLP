import spacy
from src.utils import load_text_tc

tc = load_text_tc()

# Load model and create Doc object
nlp = spacy.load('en_core_web_sm')


def find_persons(text):
    # Create Doc object
    doc = ___(___)

    # Identify the persons
    persons = [ent.____ for ent in doc.____ if ent.____ == 'PERSON']

    # Return persons
    return persons


print(____(____))