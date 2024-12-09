import re
import spacy
import pandas as pd
from unidecode import unidecode
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from spacy.lang.pt.stop_words import STOP_WORDS
import nltk

# Configurações iniciais
additional_stopwords = {"não"}
sw = list(set(stopwords.words('portuguese') + list(STOP_WORDS)))
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
def remove_stop_words(tokens, stopwords=sw):
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

    # Aplicar stemming
    df["stemmed_text"] = df["filtered_text"].apply(apply_stemming)
    df["stemmed_text"] = df["stemmed_text"].apply(lambda tokens: " ".join(tokens))

    # Aplicar lematização
    df["lemmatized_text"] = df["filtered_text"].apply(apply_lemmatization)
    df["lemmatized_text"] = df["lemmatized_text"].apply(lambda tokens: " ".join(tokens))

    # Criar DataFrame final
    processed_df = df[["clean_text", "stemmed_text", "lemmatized_text", sentiment_col]]

    return processed_df
if __name__ == "__main__":
    df = pd.read_csv("../AV1/dataset/twitter_sentiment.csv")

    processed_df = preprocess_dataset(df, text_col="tweet_text", sentiment_col="sentiment")

    # Salva o dataset processado
    processed_df.to_csv("../AV1/dataset/twitter_sentiment_processed.csv", index=False)


"""
def show_processing_steps(text):
    print(f"Texto original: {text}\n")

    # Limpeza
    cleaned_text = clean_text(text)
    print(f"Após limpeza: {cleaned_text}\n")

    # Tokenização
    tokens = word_tokenize(cleaned_text)
    print(f"Tokens: {tokens}\n")

    # Remoção de stopwords
    filtered_tokens = remove_stop_words(tokens)
    print(f"Sem stopwords: {filtered_tokens}\n")

    # Stemming
    stemmed_tokens = apply_stemming(filtered_tokens)
    print(f"Após stemming: {stemmed_tokens}\n")

    # Lematização
    lemmatized_tokens = apply_lemmatization(filtered_tokens)
    print(f"Após lematização: {lemmatized_tokens}\n")


# Exemplo de uso
text = "@NetoMoraes01 chorei que ele não ganhou. Injustiçado! :( 💗"
print(show_processing_steps(text))"""
