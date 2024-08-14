import re
import nltk
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer, WordNetLemmatizer
import numpy as np
#
# Baixar pacotes adicionais do NLTK
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('wordnet')

def to_lowercase(text):
    return text.lower()

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('portuguese'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_emoticons(text):
    emoticon_pattern = re.compile(
        r'(:\)|:\(|:D|;\)|:\]|:\[|:\*|:\||:\>|:\<|:P|:p|:\^|\):|D:|:\{|\}:|:\$|O:\)|3:\)|:-\)|:-\(|:-D|;-\)|:\'-\)|:\'-\(|:-P|:P|x-D|xD|XD|X-D|:\-O|:\-3|:\-c|:\-p|:\-\/|:\~\/|:\~\)|:\~|:>|:\<|:\<\|>|:\'\(|:\*\()',
        flags=re.UNICODE)
    return emoticon_pattern.sub(r'', text)

def remove_underscores(text):
    """
    Remove caracteres de sublinhado (_) do texto.
    """
    return text.replace('_', ' ')

def apply_stemming(text):
    stemmer = RSLPStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

def apply_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_text(text, apply_stem=False, apply_lemmatize=False):
    """
    Realiza o pré-processamento completo de um texto, com opções para stemming e lematização.
    O texto é convertido para minúsculas, menções e emojis são removidos, e stopwords são filtradas.
    Somente uma técnica de transformação (stemming ou lematização) é aplicada de cada vez.

    Args:
        text (str): Texto a ser pré-processado.
        apply_stem (bool): Se True, aplica stemming ao texto.
        apply_lemmatize (bool): Se True, aplica lematização ao texto.

    Returns:
        str: Texto pré-processado.
    """
    text = to_lowercase(text)
    text = remove_mentions(text)
    text = remove_emojis(text)
    text = remove_emoticons(text)
    text = remove_stopwords(text)
    text = remove_underscores(text)

    if apply_stem and apply_lemmatize:
        raise ValueError("Não é possível aplicar stemming e lematização simultaneamente.")

    if apply_stem:
        text = apply_stemming(text)
    elif apply_lemmatize:
        text = apply_lemmatization(text)

    return text

# Carregar o dataset
ds = load_dataset("johnidouglas/twitter-sentiment-pt-BR-md-2-l")
df = ds['train'].to_pandas()

# Verificar os valores únicos na coluna de sentimentos
print("Valores únicos em 'sentiment':", df['sentiment'].unique())

# Mostrar alguns exemplos dos rótulos
print("Exemplos de tweet e sentimento:")
print(df[['tweet_text', 'sentiment']].head())

# Selecionar 5 comentários do meio do dataset
start_index = len(df) // 2 - 5
end_index = len(df) // 2 + 5
middle_samples = df.iloc[start_index:end_index]

# Aplicar pré-processamento com stemming
texts_stemmed = [preprocess_text(text, apply_stem=True) for text in middle_samples['tweet_text']]

# Aplicar pré-processamento com lematização
texts_lemmatized = [preprocess_text(text, apply_lemmatize=True) for text in middle_samples['tweet_text']]

# Exibir os resultados
for i, (original, stemmed, lemmatized) in enumerate(zip(middle_samples['tweet_text'], texts_stemmed, texts_lemmatized)):
    print(f"\nComentário {i + 1} (Original):\n{original}")
    print(f"Comentário {i + 1} (Com Stemming):\n{stemmed}")
    print(f"Comentário {i + 1} (Com Lematização):\n{lemmatized}")
