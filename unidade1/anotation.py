import spacy
from spacy.lang.pt.examples import sentences

nlp = spacy.load("pt_core_news_sm")

for s in sentences:
    print (s, "\n")


doc = nlp(sentences[0])
print(doc.text, "\n")

#Tokenização

for token in doc:
    print(token.text)

#Lemmatização

for token in doc:
    print(token.text, token.lemma_)


#NER

for ent in doc.ents:
    print(ent.text, ent.label_)

#POS

for token in doc:
    print(token.text, token.pos_)

### Enelvo

from enelvo.normaliser import Normaliser

norm = Normaliser(tokenizer="readable")

msg = "Até hj vc n me respondeu. Oq aconteceu?"
resposta = norm.normalise(msg)
print(resposta)

# capitaliza nomes próprios
cap_pns = Normaliser(capitalize_pns=True)

# capitaliza acrônimos
cap_acs = Normaliser(capitalize_acs=True)

# capitaliza começos de frases
cap_inis = Normaliser(capitalize_inis=True)

# remove pontuação e emojis
sanitizer = Normaliser(sanitize=True)

normalizar = Normaliser(tokenizer="readable", capitalize_acs=True, 
                        capitalize_inis=True, capitalize_pns=True,
                        sanitize=True)

msg1 = "a maria foi ao shopp pq estava trsite, acho q hj foi um dia dificiu"
resposta1 = normalizar.normalise(msg1)
print(resposta1)

