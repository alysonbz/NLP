import spacy

# Carregar o modelo necessário (por exemplo, 'en_core_web_sm' para inglês)
nlp = spacy.load('en_core_web_sm')

# Criar uma instância Doc
text = 'Sundar Pichai is the CEO of Google. Its headquarters is in Mountain View.'
doc = nlp(text)

# Imprimir todas as entidades nomeadas e suas etiquetas
for ent in doc.ents:
    print(ent.text, ent.label_)
