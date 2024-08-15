import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from datasets import load_dataset

# Baixar pacotes adicionais do NLTK
nltk.download('stopwords')

# Carregar o modelo do spaCy para português
nlp = spacy.load('pt_core_news_sm')

def to_lowercase(text):
    return text.lower()

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('portuguese'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def apply_stemming(text):
    stemmer = RSLPStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

def apply_lemmatization(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def preprocess_text(text, apply_stem=False, apply_lemmatize=False):
    """
    Realiza o pré-processamento completo de um texto, com opções para stemming e lematização.
    O texto é convertido para minúsculas, menções, pontuação e números são removidos, e stopwords são filtradas.
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
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)

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

# Aplicar pré-processamento com stemming e lematização
texts_stemmed = [preprocess_text(text, apply_stem=True) for text in middle_samples['tweet_text']]
texts_lemmatized = [preprocess_text(text, apply_lemmatize=True) for text in middle_samples['tweet_text']]

# Exibir os resultados
for i, (original, stemmed, lemmatized) in enumerate(zip(middle_samples['tweet_text'], texts_stemmed, texts_lemmatized)):
    print(f"\nComentário {i + 1} (Original):\n{original}")
    print(f"Comentário {i + 1} (Com Stemming):\n{stemmed}")
    print(f"Comentário {i + 1} (Com Lematização):\n{lemmatized}")
