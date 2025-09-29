import spacy
from spacy.lang.pt.examples import sentences

nlp = spacy.load("pt_core_news_sm")

for s in sentences:
    print(f'\n {s}')

doc = nlp(sentences[0])

print("\nTokens")
for token in doc:
    print(token.text)

print("\nLemmas")
for token in doc:
    print(token.text, token.lemma_)

print("\nNER - Name Entity Recognition")
for ent in doc.ents:
    print(ent.text, ent.label_)

print("\n - Part Of Speech Tagging")
for token in doc:
    print(token.text, token.pos_)

print("FIMMM")