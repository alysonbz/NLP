# Importações
import re
import nltk
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.stem import WordNetLemmatizer

# Baixar pacotes adicionais do NLTK
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('wordnet')

# Carregar o dataset
ds = load_dataset("johnidouglas/twitter-sentiment-pt-BR-md-2-l")

# Verificar as colunas do dataset
print("Colunas disponíveis:", ds['train'].column_names)

# Verificar um exemplo dos dados
print(ds['train'][0])


# Função de pré-processamento completa
def preprocess_text(text):
    # Remover caracteres maiúsculos
    text_lower = text.lower()

    # Remover menções (@usuario)
    text_clean = re.sub(r'@\w+', '', text_lower)

    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    text_no_stopwords = ' '.join([word for word in text_clean.split() if word not in stop_words])

    # Aplicar stemming
    stemmer = RSLPStemmer()
    text_stemmed = ' '.join([stemmer.stem(word) for word in text_no_stopwords.split()])

    # Aplicar lematização
    lemmatizer = WordNetLemmatizer()
    text_lemmatized = ' '.join([lemmatizer.lemmatize(word) for word in text_no_stopwords.split()])

    return text_stemmed, text_lemmatized

# Aplicar o pré-processamento e criar colunas separadas para stemming e lematização
def map_function(example):
    stemmed, lemmatized = preprocess_text(example['tweet_text'])
    return {"tweet_stemmed": stemmed, "tweet_lemmatized": lemmatized}

# Aplicar a função de mapeamento
ds = ds.map(map_function)

# Verificar um exemplo dos dados pré-processados
print(ds['train'][0])
