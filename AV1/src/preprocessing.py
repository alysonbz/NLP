import re
import spacy
import pandas as pd
from unidecode import unidecode
from string import punctuation
from nltk.corpus import stopwords
from spacy.lang.pt.stop_words import STOP_WORDS
import nltk

# Configurações iniciais
nltk.download('stopwords')
nltk.download('punkt')
sw = list(set(stopwords.words('portuguese') + list(STOP_WORDS)))
nlp = spacy.load("pt_core_news_sm")

stopwords_pt = [
    "a", "as", "o", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "com", "para", "e", "é", "ser", "ter", "se", "que", "ou", "como", "foi", "há", "onde", "qual", "porque",
    "está", "estão", "este", "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "meu", "min", "meus",
    "minhas", "seu", "sua", "seus", "uns", ]

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
def remove_stop_words(tokens, stopwords=sw):
    """voce pode escolher qual lista de stop words usar"""
    return [word for word in tokens if word not in stopwords_pt]


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
    df["tokenized_text"] = df["clean_text"].apply(lambda text: text.split())


    # Remover stopwords
    df["filtered_text"] = df["tokenized_text"].apply(remove_stop_words)
    df["clean_text"] = df["filtered_text"].apply(lambda tokens: " ".join(tokens))

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
    df = pd.read_csv("../../AV2/subs/portuguese_hate_.csv")

    # Processar dados
    processed_df = preprocess_dataset(df, text_col="text", sentiment_col="is_hate_speech")

    # Salvar dataset processado
    processed_df.to_csv("portuguese_hate_processed_stopwords_manual.csv", index=False)

    print(processed_df.head())
