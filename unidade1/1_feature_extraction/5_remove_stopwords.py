import spacy
from src.utils import load_gettyburg

# Carregar o texto de Gettysburg
gettysburg = load_gettyburg()

# Carregar o modelo do spaCy para inglês ('en_core_web_sm' é um modelo pequeno para inglês)
nlp = spacy.load('en_core_web_sm')

# Obter as stopwords do modelo do spaCy
stopwords = spacy.lang.en.stop_words.STOP_WORDS

# Processar o texto usando o modelo carregado
doc = nlp(gettysburg)

# Gerar lemas (lemmatized tokens)
lemmas = [token.lemma_ for token in doc]

# Remover stopwords e tokens não alfabéticos
a_lemmas = [lemma for lemma in lemmas
            if lemma.isalpha() and lemma not in stopwords]

# Imprimir a string após a limpeza do texto
print(' '.join(a_lemmas))
