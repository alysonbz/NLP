import spacy
from spacy.lang.pt.examples import sentences

# Parte 1
"""nlp = spacy.load('en_core_web_sm')"""

for s in sentences:
    print(s, '\n')

# Parte 2
nlp =spacy.load('pt_core_news_sm')
doc = nlp(sentences[0])
print(doc.text)

# Tokenização
for token in doc:
    print(token.text)

print("_________________________________________")

# Lemmatização
for token in doc:
    print(token.text, token.lemma_)

print("_________________________________________")


# NER
for ent in doc.ents:
    print(ent.text, ent.label_)

print("_________________________________________")

# POS
for token in doc:
    print(token.text, token.pos_)


print("_________________________________________")

# SIGNIFICADO DE TAGGINGS

# exemplo
texto = "O rato roeu a roupa do rei de Roma."

# Processa o texto
doc = nlp(texto)

print("=== Análise de Classes Gramaticais (POS Tagging) ===\n")
for token in doc:
    print(f"{token.text:10} → {token.pos_}")

#explicação das tags
print("\n=== Significado das Tags ===")
print("DET   → Determinante (artigos, pronomes demonstrativos etc.)")
print("NOUN  → Substantivo comum")
print("PROPN → Substantivo próprio (nomes de pessoas, lugares etc.)")
print("VERB  → Verbo (ação ou estado)")
print("ADP   → Preposição (como 'de', 'em', 'com')")
print("PUNCT → Pontuação (., !, ?)")
