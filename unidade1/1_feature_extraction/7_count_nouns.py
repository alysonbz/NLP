import spacy
nlp = spacy.load('en_core_web_sm')


# Returns number of proper nouns
def proper_nouns(text, model=nlp):
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]

    # Return number of proper nouns
    return len([p for p in pos if p == "PROPN"])


print(proper_nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))