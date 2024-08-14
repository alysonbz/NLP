import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

stop_words = set(stopwords.words('portuguese'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Função de pré-processamento básico
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove caracteres não alfanuméricos
    text = re.sub(r'\s+', ' ', text)  # Remove espaços múltiplos
    text = text.lower().strip()  # Converte para minúsculas e remove espaços nas extremidades
    return text

# Função para remover stopwords
def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Função para aplicar stemming
def stem_text(text):
    words = text.split()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Função para aplicar lemmatization
def lemmatize_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)
