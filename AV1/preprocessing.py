import re
import spacy
from pandas import read_csv
from unidecode import unidecode
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spacy.lang.pt.stop_words import STOP_WORDS
import nltk

# Configurações iniciais
nltk.download('stopwords')
nltk.download('punkt')
# Lista de stopwords personalizada, mantendo o "não"
default_stopwords = set(stopwords.words('portuguese')) | STOP_WORDS
custom_stopwords = default_stopwords - {"não"}
nlp = spacy.load("pt_core_news_sm")

# Função para limpar o texto
def clean_text(text):
    text = text.lower()  # Caixa baixa
    text = re.sub('@[^\s]+', '', text)
    text = unidecode(text)  # Remoção de acentos
    text = re.sub('<[^<]+?>', '', text)
    text = ''.join(c for c in text if not c.isdigit())
    text = re.sub(r'(www\.[^\s]+|https?://[^\s]+)', '', text)
    text = ''.join(c for c in text if c not in punctuation)
    return text


# Função para remover stopwords
def remove_stop_words(tokens, stopwords=custom_stopwords):
    return [word for word in tokens if word not in stopwords]


# Função para aplicar stemming
def apply_stemming(tokens):
    stemmer = nltk.stem.SnowballStemmer("portuguese")
    return [stemmer.stem(word) for word in tokens]


# Função para aplicar lematização
def apply_lemmatization(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]


# Função principal de pré-processamento
def preprocess_dataset(df, text_col, sentiment_col):
    # Limpar textos
    df["clean_text"] = df[text_col].apply(clean_text)

    # Tokenizar textos
    df["tokenized_text"] = df["clean_text"].apply(word_tokenize)

    # Remover stopwords
    df["filtered_text"] = df["tokenized_text"].apply(remove_stop_words)
    df["clean_text"] = df["filtered_text"].apply(lambda tokens: " ".join(tokens))

    # Reclassificar coluna rating
    df[sentiment_col] = df[sentiment_col].apply(
        lambda x: 0 if x == 1 else (1 if x in [2, 3] else 2)
    )

    # Aplicar stemming
    df["stemmed_text"] = df["filtered_text"].apply(apply_stemming)
    df["stemmed_text_complete"] = df["stemmed_text"].apply(lambda tokens: " ".join(tokens))

    # Aplicar lematização
    df["lemmatized_text"] = df["filtered_text"].apply(apply_lemmatization)
    df["lemmatized_text_complete"] = df["lemmatized_text"].apply(lambda tokens: " ".join(tokens))

    # Criar DataFrame final
    processed_df = df[
        ["clean_text", "stemmed_text_complete", "lemmatized_text_complete", sentiment_col]]

    return processed_df


if __name__ == "__main__":
    # Carregar dataset
    df = read_csv("../AV1/Dataset/b2w_novo.csv")

    # Processar dados
    processed_df = preprocess_dataset(df, text_col="review_text", sentiment_col="rating")

    # Salvar dataset processado
    processed_df.to_csv("b2w_processed.csv", index=False)

    print(processed_df.head())
