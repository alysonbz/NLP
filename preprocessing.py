"""import pandas as pd

df = pd.read_parquet("train-00000-of-00001.parquet")
print(df.info())

print(df.head(10))

print(df['relatedness_score'].describe())
print(df['entailment_judgment'].value_counts())
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer, WordNetLemmatizer

# ===============================================================
# 2. Configurações gerais
# ===============================================================
stop_words = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()
lemmatizer = WordNetLemmatizer()  # funciona para pt, mas limitado


# ===============================================================
# 3. Funções de pré-processamento
# ===============================================================

def to_lower(text: str) -> str:
    """Converte para minúsculas."""
    return text.lower()


def remove_mentions(text: str) -> str:
    """Remove menções (@usuario)."""
    return re.sub(r"@\w+", "", text)


def remove_urls(text: str) -> str:
    """Remove URLs completas."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_punctuation(text: str) -> str:
    """Remove pontuações, mantendo apenas letras e espaços."""
    return re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)


def tokenize(text: str) -> list:
    """Tokeniza por espaços."""
    return text.split()


def remove_stopwords(tokens: list) -> list:
    """Remove stopwords da lista de tokens."""
    return [t for t in tokens if t not in stop_words]


def apply_stemming(tokens: list) -> list:
    """Aplica stemming com RSLP (ótimo para português)."""
    return [stemmer.stem(t) for t in tokens]


def apply_lemmatization(tokens: list) -> list:
    """Aplica lematização (limitada para pt)."""
    return [lemmatizer.lemmatize(t) for t in tokens]


def detokenize(tokens: list) -> str:
    """Une tokens de volta em texto."""
    return " ".join(tokens)


# ===============================================================
# 4. Pipeline de pré-processamento para uma sentença
# ===============================================================

def preprocess_sentence(text: str,
                        use_stemming=False,
                        use_lemmatization=True,
                        remove_stops=True):
    """Pré-processa uma sentença individual."""

    text = to_lower(text)
    text = remove_mentions(text)
    text = remove_urls(text)
    text = remove_punctuation(text)

    tokens = tokenize(text)

    if remove_stops:
        tokens = remove_stopwords(tokens)

    if use_stemming:
        tokens = apply_stemming(tokens)

    if use_lemmatization:
        tokens = apply_lemmatization(tokens)

    return detokenize(tokens)


# ===============================================================
# 5. Função principal para preprocessar o dataset
# ===============================================================

def preprocess_dataset(df: pd.DataFrame,
                       use_stemming=False,
                       use_lemmatization=True,
                       remove_stops=True):
    """
    Aplica o pré-processamento nas colunas 'premise' e 'hypothesis'.
    Retorna um novo dataframe preprocessado.
    """
    df = df.copy()

    df["premise_clean"] = df["premise"].apply(
        lambda x: preprocess_sentence(
            x,
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization,
            remove_stops=remove_stops
        )
    )

    df["hypothesis_clean"] = df["hypothesis"].apply(
        lambda x: preprocess_sentence(
            x,
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization,
            remove_stops=remove_stops
        )
    )

    return df


# ===============================================================
# 6. Execução manual para testar (opcional)
# ===============================================================
if __name__ == "__main__":
    df = pd.read_parquet("train-00000-of-00001.parquet")
    df_clean = preprocess_dataset(df)
    print(df_clean.head())

