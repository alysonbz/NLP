import spacy
import pandas as pd
from nltk.stem import PorterStemmer


# Carregar o modelo de linguagem do spacy
nlp = spacy.load("en_core_web_sm")

# Texto de exemplo
text = "SpaCy is an open-source software library for advanced NLP."

# Processar o texto com o modelo
doc = nlp(text)

# Criar listas para armazenar os dados
words = []
pos_tags = []
lemmas = []

# Extrair palavras, POS tags e lemas
for token in doc:
    words.append(token.text)
    pos_tags.append(token.pos_)
    lemmas.append(token.lemma_)

# Criar DataFrame com pandas
df = pd.DataFrame({
    'Word': words,
    'POS Tag': pos_tags,
    'Lemma': lemmas
})

# Mostrar a tabela
print(df.to_string())

################################################################################################################
#Lematização e Stemming

stemmer = PorterStemmer()

# Texto de exemplo
text = "running runs runner cared caring cares"

# Processar texto com Spacy para Lematização
doc = nlp(text)

# Listas para armazenar palavras, lemas e stemming
words = []
lemmas = []
stems = []

for token in doc:
    words.append(token.text)
    lemmas.append(token.lemma_)
    stems.append(stemmer.stem(token.text))

# Criar DataFrame para exibir os resultados
df = pd.DataFrame({
    'Word': words,
    'Lemma': lemmas,
    'Stem': stems
})

# Mostrar a tabela no Colab
print(df)
