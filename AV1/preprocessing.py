import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import re

# Baixar dados necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Função para carregar dados
def load_data(file_path):
    return pd.read_csv(file_path)

# Função para remover caracteres especiais e transformar em minúsculas
def clean_text(text):
    text = text.lower()  # Remove maiúsculas
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação
    return text

# Função para remoção de stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_text = [word for word in words if word not in stop_words]
    return ' '.join(filtered_text)

# Função de stemming
def stem_text(text):
    stemmer = SnowballStemmer('english')
    words = nltk.word_tokenize(text)
    stemmed_text = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_text)

# Função de lematização
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_text)

# Função para identificar menções (@)
def identify_mentions(text):
    mentions = re.findall(r'@\w+', text)
    return mentions

# Função de pré-processamento completo
def preprocess_text(df, text_column):
    df[text_column] = df[text_column].apply(clean_text)
    df[text_column] = df[text_column].apply(remove_stopwords)
    df[text_column] = df[text_column].apply(stem_text)
    return df

# Exemplo de uso
train_df = load_data('train.csv')
train_df = preprocess_text(train_df, 'essay')