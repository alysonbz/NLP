import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Inicializar o stemmer e lemmatizer
stemmer = RSLPStemmer()
lemmatizer = WordNetLemmatizer()

# Carregar stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Função para converter o texto para minúsculas
def to_lowercase(text):
    return text.lower()

# Função para remover caracteres especiais e números
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s@]', '', text)  # Mantém "@" para verificação de menções

# Função para remover stopwords
def remove_stopwords(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Função para aplicar stemming
def apply_stemming(text):
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

# Função para aplicar lematização
def apply_lemmatization(text):
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

# Função para verificar menções (@usuário)
def check_mentions(text):
    mentions = re.findall(r'@\w+', text)
    return mentions if mentions else None

# Função principal de pré-processamento, que aplica todas as etapas
def preprocess_text(text, apply_lemma=False):
    # Converter para minúsculas
    text = to_lowercase(text)

    # Remover caracteres especiais, mantendo @ para menções
    text = remove_special_characters(text)

    # Verificar menções
    mentions = check_mentions(text)

    # Remover stopwords
    text = remove_stopwords(text)

    # Aplicar lematização ou stemming
    if apply_lemma:
        text = apply_lemmatization(text)
    else:
        text = apply_stemming(text)

    return text, mentions
# test_preprocessing.py
import pandas as pd
from preprocessing import preprocess_text

# Carregar os datasets
fake_news_path = "C:/Users/bianc/OneDrive/Documentos/NLP/AV1/News_fake.csv"
not_fake_news_path = "C:/Users/bianc/OneDrive/Documentos/NLP/AV1/News_notFake.csv"

df_fake = pd.read_csv(fake_news_path)
df_not_fake = pd.read_csv(not_fake_news_path)

# Verifique se os datasets foram carregados corretamente
if df_fake.empty or df_not_fake.empty:
    raise ValueError("Um ou ambos os datasets estão vazios. Verifique o caminho dos arquivos.")

# Unir os datasets e criar rótulos
df_fake['label'] = 1
df_not_fake['label'] = 0
df = pd.concat([df_fake, df_not_fake])

# Verificar se a coluna 'title' está presente
if 'title' not in df.columns:
    raise ValueError("A coluna 'title' não foi encontrada no dataset.")

# Exemplo de uso em um dataset de notícias
for text in df['title']:
    processed_text, mentions = preprocess_text(text)
    print(f"Texto Processado: {processed_text}")
    if mentions:
        print(f"Menções encontradas: {mentions}")
