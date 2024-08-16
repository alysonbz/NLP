# Packages -------------------------------------------------------------------------------------------------------------

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
import spacy


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('all')

stemmer = RSLPStemmer()
stop_words = set(stopwords.words('portuguese'))
nlp = spacy.load('pt_core_news_sm')


# Função de pré-processamento do texto ---------------------------------------------------------------------------------

def preprocessar_texto(texto):
    texto = re.sub(r'(@\w+|#\w+|http\S+|[^a-zA-Z\s])', '', texto)  # remover menções, hashtags, URLs e deixando apenas textos e espaços
    texto = ' '.join([plv for plv in word_tokenize(texto) if plv not in stop_words])  # remover stopwords
    texto = ' '.join([token.lemma_ for token in nlp(texto)])  # lematizar
    # texto = ' '.join([stemmer.stem(plv) for plv in word_tokenize(texto)])  # stemming
    texto = texto.lower()  # converter tudo em caracteres minúsculos
    return texto
