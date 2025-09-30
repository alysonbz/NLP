# pip install spacy                             instalar a biblioteca
# python -m spacy download pt_core_news_sm      instalar o modelo em português
# nlp = spacy.load("pt_core_news_sm")           carregar o modelo pré-treinado

import spacy
nlp = spacy.load("pt_core_news_sm")
from spacy.lang.pt.examples import sentences

for s in sentences:
    print('\nSentença: ', s, '\n')

# Criando o objeto spacy
doc = nlp(sentences[0])
print(doc.text)

# Tokentização
print("\nTOKENTIZAÇÃO\n")
for token in doc:
    print(token.text)

# Lemmatização
print("\nLEMMATIZAÇÃO\n")
for token in doc:
    print(token.text, token.lemma_)

# NER - Name Entity Recognition
print("\nNER\n")
for ent in doc.ents:
    print(ent.text, ent.label_)

# POS - Part-Of-Speech (tagging)
print("\nPOS\n")
for token in doc:
    print(token.text, token.pos_)


print('\nO significado de POS é o processo de rotular cada palavra de um texto com sua classe gramatical (substantivo, verbo, ...)\n')
print('PROPN (Proper Noun) - Substantivo próprio\nAUX (Auxiliary verb) - Verbo auxiliar\nVERB (Verb) - Verbo\nDET (Determiner) - Determinante / artigo / pronome determinante\nNOUN (Noun) - Substantivo comum\nADV (Adverb) - Advérbio\nADP (Adposition) - Preposição ou posposição\nNUM (Numeral) - Número ou numeral\nSYN (Symbol) - Símbolo')
