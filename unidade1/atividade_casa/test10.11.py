import spacy 
nlp = spacy.load("en_core_web_sm")

string = "Hello! I don't know what I'm doing here."
doc= nlp(string)
tokens = [token.text for token in doc]
print(tokens)

lemma = [token.lemma_ for token in doc]
print(lemma)

pos = [(token.text , token.pos_ )for token in doc]
print(pos)

string1 = "Jonh Doe is a software working at Google. He lives in France"
doc1= nlp(string1)
ne = [(ent.text, ent.label_) for ent in doc1.ents]
print(ne)