import nltk
import spacy
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

# Baixar pacotes NLTK necessários
nltk.download('punkt')
nltk.download('stopwords')

# Carregar o modelo spaCy para lematização
nlp = spacy.load('pt_core_news_sm')

# Instanciar o corretor ortográfico
spell = SpellChecker(language='pt')

# Função para corrigir erros ortográficos usando pyspellchecker
def corrigir_ortografia(texto):
    palavras = texto.split()
    palavras_corrigidas = [spell.candidates(palavra).pop() if spell.candidates(palavra) else palavra for palavra in palavras]
    return ' '.join(palavras_corrigidas)

# Texto com erros gramaticais
texto = 'A maternidadI é o melhor momentu da vida de uma mulher.'

# a) Realizar a correção ortográfica
texto_corrigido = corrigir_ortografia(texto)
print('Texto original:', texto)
print('Texto corrigido:', texto_corrigido)

# b) Tokenizar o texto
tokens = word_tokenize(texto_corrigido, language='portuguese')

# c) Remover as stop words
stop_words = set(stopwords.words('portuguese'))
tokens_sem_stopwords = [token for token in tokens if token.lower() not in stop_words]

# d) Criar a tabela com token, stemming e lematização
# Stemming
stemmer = PorterStemmer()
tokens_stemming = [stemmer.stem(token) for token in tokens_sem_stopwords]

# Lematização
tokens_lemmas = [nlp(token)[0].lemma_ for token in tokens_sem_stopwords]

# Criar o DataFrame
df = pd.DataFrame({
    'Token': tokens_sem_stopwords,
    'Stemming': tokens_stemming,
    'Lematização': tokens_lemmas
})

print('\nTabela de Tokens, Stemming e Lematização:')
print(df)
