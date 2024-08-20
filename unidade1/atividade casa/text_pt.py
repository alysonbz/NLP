# Faça uma implementação para que sua aplicação receba um texto com erros gramaticais em português:
# a) Realize a correção do texto
# b) Tokenize o texto
# c) Remova as stop words
# d) Faça uma tabela mostrando 3 colunas para todos os tokens: token , stemming e lematização


import pandas as pd
import language_tool_python
import spacy
from nltk.stem import RSLPStemmer
from spacy.lang.pt.stop_words import STOP_WORDS

# a) Realize a correção do texto
tool = language_tool_python.LanguageTool('pt-BR')

texto = """O desenvolvimento sustentável é uma abordagem que busca equilibrar o crescimento econômico, a proteção ambiental e o bem-estar social. 
Esse conceito envolve a utilização responsável dos recursos naturais, garantindo que as necessidades das gerações atuais sejam atendidas sem comprometer as futuras. 
Promover a sustentabilidade é essencial para preservar o meio ambiente e assegurar uma qualidade de vida adequada para todos."""

texto_corrigido = tool.correct(texto)
print("Texto corrigido:")
print(texto_corrigido)

# b) Tokenize o texto usando spaCy
nlp = spacy.load('pt_core_news_sm')
doc = nlp(texto_corrigido)

tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
print("Tokens:")
print(tokens)

# c) Remova as stop words
stop_words = set(STOP_WORDS)
tokens_sem_stopwords = [token for token in tokens if token.lower() not in stop_words]
print("Tokens sem stop words:")
print(tokens_sem_stopwords)

# Stemming com RSLPStemmer para português
from nltk import download
download('rslp')  # Baixa o recurso RSLP do NLTK

stemmer = RSLPStemmer()
stems = [stemmer.stem(token) for token in tokens_sem_stopwords]
print("Stemming:")
print(stems)

# Lematização com spaCy
lemmas = [token.lemma_ for token in doc if token.text in tokens_sem_stopwords]
print("Lemmatização:")
print(lemmas)

# d) Faça uma tabela mostrando 3 colunas para todos os tokens: token, stemming e lematização
# Garantir que a lista de lemas tenha o mesmo comprimento que a lista de tokens_sem_stopwords

# Criar um mapa de lemas
lemma_map = {token.text: token.lemma_ for token in doc}
data = {
    'Token': tokens_sem_stopwords,
    'Stemming': [stemmer.stem(token) for token in tokens_sem_stopwords],
    'Lematização': [lemma_map.get(token, token) for token in tokens_sem_stopwords]  # Usa o token original se o lema não for encontrado
}

df = pd.DataFrame(data)
print(df)
