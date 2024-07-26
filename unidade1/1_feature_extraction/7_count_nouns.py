import spacy

nlp = spacy.load('en_core_web_sm')


# Returns number of proper nouns
def proper_nouns(text, model=nlp):
    # Create doc object
    doc = model(text)

    # Initialize a counter for proper nouns
    num_proper_nouns = 0

    # Iterate over each token in the doc
    for token in doc:
        # Check if the token is a proper noun (PROPN)
        if token.pos_ == 'PROPN':
            num_proper_nouns += 1

    # Return the count of proper nouns
    return num_proper_nouns


print(proper_nouns("Abdul, Bill and Cathy went to the market to buy apples.", nlp))
