import spacy
from collections import Counter

# Carregar o modelo de linguagem em português
nlp = spacy.load('pt_core_news_sm')


# 1. Analisando um texto básico
def analyze_text_basic(text):
    doc = nlp(text)
    print("1. Tokens no texto:")
    for token in doc:
        print(token.text)


# 2. Identificando as entidades nomeadas (NER)
def identify_entities(text):
    doc = nlp(text)
    print("\n2. Entidades Nomeadas (NER):")
    for ent in doc.ents:
        print(ent.text, ent.label_)


# 3. Análise de dependência (relações entre palavras)
def analyze_dependency(text):
    doc = nlp(text)
    print("\n3. Análise de Dependência:")
    for token in doc:
        print(token.text, token.dep_, token.head.text)


# 4. Lematização (reduzir palavras à sua forma básica)
def lemmatization(text):
    doc = nlp(text)
    print("\n4. Lematização:")
    for token in doc:
        print(f'{token.text} -> {token.lemma_}')


# 5. Identificar as partes do discurso (POS tagging)
def pos_tagging(text):
    doc = nlp(text)
    print("\n5. Partes do Discurso (POS tagging):")
    for token in doc:
        print(token.text, token.pos_)


# 6. Tokenização e extração de frases
def sentence_extraction(text):
    doc = nlp(text)
    print("\n6. Tokenização e Extração de Frases:")
    for sent in doc.sents:
        print(f'Frase: {sent.text}')
        for token in sent:
            print(f'  - {token.text}')


# 7. Similaridade entre dois textos
def text_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity = doc1.similarity(doc2)
    print(f"\n7. Similaridade entre os textos: {similarity:.2f}")


if __name__ == "__main__":
    # Definindo textos para as análises
    text1 = "O gato está em cima do telhado."
    text2 = "O cachorro está no telhado."

    # Chamar as funções de análise
    analyze_text_basic(text1)
    identify_entities("A empresa Google foi fundada por Larry Page e Sergey Brin em 1998.")
    analyze_dependency(text1)
    lemmatization("Os gatos estão correndo e comendo.")
    pos_tagging("O cachorro corre rápido.")
    sentence_extraction("O sol está brilhando. Vamos sair para aproveitar o dia!")
    text_similarity(text1, text2)
