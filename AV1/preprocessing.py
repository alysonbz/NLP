from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

try:
    import spacy
    _NLP_PT = spacy.load("pt_core_news_sm")
except Exception:
    _NLP_PT = None  

def load_binary_hatespeech_dataset(
    csv_path: str,
    text_col: str = "text",
    label_col: str = "hatespeech_comb",
) -> Tuple[pd.Series, pd.Series]:

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col])
    X = df[text_col].astype(str)
    y = df[label_col].astype(int)
    return X, y

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
NUM_PATTERN = re.compile(r"\d+")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(
    text: str,
    lower: bool = True,
    remove_urls: bool = True,
    remove_mentions_flag: bool = True,
    remove_hashtags: bool = False,
    remove_numbers: bool = True,
    remove_punctuation: bool = True,
) -> str:

    if not isinstance(text, str):
        text = str(text)

    if lower:
        text = text.lower()

    if remove_urls:
        text = URL_PATTERN.sub(" ", text)

    if remove_mentions_flag:
        text = MENTION_PATTERN.sub(" ", text)
    else:
        text = MENTION_PATTERN.sub(" @user ", text)

    if remove_hashtags:
        text = re.sub(r"#", " ", text)

    if remove_numbers:
        text = NUM_PATTERN.sub(" ", text)

    if remove_punctuation:
        text = re.sub(r"[^\w\sáéíóúâêôãõç]", " ", text, flags=re.UNICODE)

    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    return text

try:
    _ = stopwords.words("portuguese")
except LookupError:
    nltk.download("stopwords")

PORTUGUESE_STOPWORDS = set(stopwords.words("portuguese"))


def tokenize(text: str) -> List[str]:

    return text.split()


def remove_stopwords(
    tokens: List[str],
    extra_stopwords: Optional[List[str]] = None,
) -> List[str]:

    stop_set = set(PORTUGUESE_STOPWORDS)
    if extra_stopwords:
        stop_set.update(extra_stopwords)

    return [t for t in tokens if t not in stop_set]

_STEMMER_PT = RSLPStemmer()


def stem_tokens(tokens: List[str]) -> List[str]:

    return [_STEMMER_PT.stem(t) for t in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:

    if _NLP_PT is None:
        return tokens

    doc = _NLP_PT(" ".join(tokens))
    return [token.lemma_ for token in doc]

def preprocess_text(
    text: str,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
    remove_stop: bool = True,
    extra_stopwords: Optional[List[str]] = None,
    clean_kwargs: Optional[Dict[str, Any]] = None,
) -> str:

    if clean_kwargs is None:
        clean_kwargs = {}
    cleaned = clean_text(text, **clean_kwargs)
    tokens = tokenize(cleaned)

    if remove_stop:
        tokens = remove_stopwords(tokens, extra_stopwords=extra_stopwords)

    if use_stemming and use_lemmatization:
        raise ValueError("Escolha apenas uma opção: stemming OU lemmatização.")

    if use_stemming:
        tokens = stem_tokens(tokens)
    elif use_lemmatization:
        tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "hatespeech_comb",
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], pd.Series]:

    if preprocess_kwargs is None:
        preprocess_kwargs = {}

    df = df.dropna(subset=[text_col, label_col])

    X_raw = df[text_col].astype(str)
    y = df[label_col].astype(int)

    X_processed = [
        preprocess_text(text, **preprocess_kwargs) for text in X_raw
    ]

    return X_processed, y

def contar_tokens(lista_textos):
    return [len(t.split()) for t in lista_textos]

def gerar_graficos_preprocessamento(csv_path: str):
    X_raw, y = load_binary_hatespeech_dataset(csv_path)

    X_lemma = [preprocess_text(t, use_lemmatization=True, use_stemming=False) 
               for t in X_raw]

    X_stem = [preprocess_text(t, use_stemming=True, use_lemmatization=False) 
              for t in X_raw]

    raw_tokens = contar_tokens(X_raw)
    lemma_tokens = contar_tokens(X_lemma)
    stem_tokens_c = contar_tokens(X_stem)

    plt.figure(figsize=(8,5))
    plt.hist(raw_tokens, bins=50, alpha=0.6, label="Original")
    plt.hist(lemma_tokens, bins=50, alpha=0.6, label="Lematização")
    plt.hist(stem_tokens_c, bins=50, alpha=0.6, label="Stemming")
    
    plt.title("Distribuição do Número de Tokens (Antes e Depois do Pré-processamento)")
    plt.xlabel("Quantidade de Tokens")
    plt.ylabel("Frequência")
    plt.legend()
    plt.tight_layout()
    plt.savefig("preprocess_tokens_hist.png", dpi=300)
    plt.close()

    print("✔ Gráfico salvo: preprocess_tokens_hist.png")

    medias = [
        np.mean(raw_tokens),
        np.mean(lemma_tokens),
        np.mean(stem_tokens_c)
    ]

    plt.figure(figsize=(6,4))
    plt.bar(["Original", "Lematização", "Stemming"], medias, color=["#4C72B0","#55A868","#C44E52"])
    plt.title("Média de Tokens por Documento")
    plt.ylabel("Média de tokens")
    plt.tight_layout()
    plt.savefig("preprocess_media_tokens.png", dpi=300)
    plt.close()

    print("✔ Gráfico salvo: preprocess_media_tokens.png")

    raw_words = " ".join(X_raw).split()
    lemma_words = " ".join(X_lemma).split()
    stem_words = " ".join(X_stem).split()

    top_raw = Counter(raw_words).most_common(15)
    top_lemma = Counter(lemma_words).most_common(15)
    top_stem = Counter(stem_words).most_common(15)

    def plot_top(words_freq, title, filename):
        words, counts = zip(*words_freq)
        plt.figure(figsize=(8,4))
        plt.bar(words, counts, color="#4C72B0")
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✔ Gráfico salvo: {filename}")

    plot_top(top_raw, "Top 15 Palavras (Original)", "top_words_original.png")
    plot_top(top_lemma, "Top 15 Palavras (Lematização)", "top_words_lemmatization.png")
    plot_top(top_stem, "Top 15 Palavras (Stemming)", "top_words_stemming.png")

    print("\nTodos os gráficos da etapa 1 foram gerados com sucesso.")

if __name__ == "__main__":
    gerar_graficos_preprocessamento("2019-05-28_portuguese_hate_speech_binary_classification.csv")