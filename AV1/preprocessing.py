import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer, WordNetLemmatizer

# recusos do NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('rslp')

# Função para converter texto para minúsculas
def to_lowercase(text):
    if isinstance(text, str):  # Verificar se o texto é uma string
        return text.lower()
    return text

# Função para remover pontuações
def remove_punctuation(text):
    if isinstance(text, str):  # Verificar se o texto é uma string
        return text.translate(str.maketrans('', '', string.punctuation))
    return text

# Função para remover números
def remove_numbers(text):
    if isinstance(text, str):  # Verificar se o texto é uma string
        return re.sub(r'\d+', '', text)
    return text

# Função para remover stopwords
def remove_stopwords(text):
    if isinstance(text, str):  # Verificar se o texto é uma string
        stop_words = set(stopwords.words('portuguese'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return ' '.join(filtered_text)
    return text

# Função para realizar stemming
def perform_stemming(text):
    if isinstance(text, str):  # Verificar se o texto é uma string
        stemmer = RSLPStemmer()
        word_tokens = word_tokenize(text)
        stemmed_text = [stemmer.stem(word) for word in word_tokens]
        return ' '.join(stemmed_text)
    return text

# Função para realizar lemmatização
def perform_lemmatization(text):
    if isinstance(text, str):  # Verificar se o texto é uma string
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text)
        lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lemmatized_text)
    return text

# Função para verificar e remover menções (@username)
def remove_mentions(text):
    if isinstance(text, str):  # Verificar se o texto é uma string
        return re.sub(r'@\w+', '', text)
    return text

# Função principal para pré-processar o texto
def preprocess_text(text, use_stemming=True, use_lemmatization=False):
    text = to_lowercase(text)
    text = remove_mentions(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)

    if use_lemmatization:
        text = perform_lemmatization(text)
    elif use_stemming:
        text = perform_stemming(text)

    return text
