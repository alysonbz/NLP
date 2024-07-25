import spacy
from src.utils import load_text_tc

# Carregar o texto usando a função load_text_tc() (supondo que ela carregue o texto desejado)
tc = load_text_tc()

# Carregar o modelo en_core_web_sm do spaCy
nlp = spacy.load('en_core_web_sm')

def find_persons(text):
    # Criar um objeto Doc
    doc = nlp(text)

    # Identificar as pessoas (entidades do tipo 'PERSON')
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

    # Retornar as pessoas identificadas
    return persons

# Chamar a função find_persons com o texto carregado (tc)
print(find_persons(tc))
