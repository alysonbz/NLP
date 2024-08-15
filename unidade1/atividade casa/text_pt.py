import pandas as pd
import language_tool_python
import spacy
from nltk.stem import RSLPStemmer
from spacy.lang.pt.stop_words import STOP_WORDS

# a) Realize a correção do texto
tool = language_tool_python.LanguageTool('pt-BR')

texto = """A vida com python é [insira algo aqui], É uma linguagem com tipificação fraca,
 mas o python é de alto nível, isso significa que está mais próxima da linguagem humama. 
 A linguagem python se chama assim por causa de um programa que o seu criador gostava. Não sei mais o que escrever."""

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

data = {
    'Token': tokens_sem_stopwords,
    'Stemming': stems
}

lemma_map = {token.text: token.lemma_ for token in doc}
data['Lematização'] = [lemma_map.get(token, None) for token in tokens_sem_stopwords]

df = pd.DataFrame(data)
print(df)
