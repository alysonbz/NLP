from atribute_extraction import FeatureExtractor
from preprocessing import preprocess_dataset
import pandas as pd

class ClassificationComparator:

    def __init__(self, df, text_column, label_column):
        self.df = df.reset_index(drop=True)
        self.text_column = text_column
        self.label_column = label_column
        self.results = []

    def evaluate(self, X, y, feature_name, preprocessing_flag):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        clf = LogisticRegression(max_iter=300)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        self.results.append({
            "Features": feature_name,
            "Pré-processado": preprocessing_flag,
            "Acurácia": acc
        })

    def run_all(self):
        y_raw = self.df[self.label_column].reset_index(drop=True)

        # ---------- SEM PRÉ-PROCESSAMENTO ----------
        texts_raw = self.df[self.text_column].astype(str).tolist()
        fe_raw = FeatureExtractor(texts_raw)

        bow, _ = fe_raw.bag_of_words()
        self.evaluate(bow, y_raw, "Bag of Words", "Sem")

        tfidf, _ = fe_raw.tfidf()
        self.evaluate(tfidf, y_raw, "TF-IDF", "Sem")

        cooc, _ = fe_raw.cooccurrence_matrix()
        self.evaluate(cooc, y_raw, "Coocorrência", "Sem")

        w2v, _ = fe_raw.word2vec()
        self.evaluate(w2v, y_raw, "Word2Vec", "Sem")

        stats = fe_raw.statistical_analysis()
        self.evaluate(stats, y_raw, "Estatísticas", "Sem")

        # ---------- COM PRÉ-PROCESSAMENTO ----------
        df_clean = preprocess_dataset(self.df).reset_index(drop=True)
        y_clean = df_clean[self.label_column].reset_index(drop=True)

        texts_clean = df_clean["headline_clean"].tolist()
        fe_clean = FeatureExtractor(texts_clean)

        bow, _ = fe_clean.bag_of_words()
        self.evaluate(bow, y_clean, "Bag of Words", "Com")

        tfidf, _ = fe_clean.tfidf()
        self.evaluate(tfidf, y_clean, "TF-IDF", "Com")

        cooc, _ = fe_clean.cooccurrence_matrix()
        self.evaluate(cooc, y_clean, "Coocorrência", "Com")

        w2v, _ = fe_clean.word2vec()
        self.evaluate(w2v, y_clean, "Word2Vec", "Com")

        stats = fe_clean.statistical_analysis()
        self.evaluate(stats, y_clean, "Estatísticas", "Com")

        return pd.DataFrame(self.results)
    
# from classification_compare import ClassificationComparator

df = pd.read_csv("brazilian_headlines_sentiments.csv")

# Converter score p/ número real
df["sentimentScorePortuguese"] = pd.to_numeric(
    df["sentimentScorePortuguese"], errors="coerce"
)

# Remover NaN no texto ou score
df = df.dropna(subset=["headlinePortuguese", "sentimentScorePortuguese"])

# Remover textos vazios
df["headlinePortuguese"] = df["headlinePortuguese"].astype(str)
df = df[df["headlinePortuguese"].str.strip() != ""]

# Criar o label discreto
def score_to_label(score):
    if score <= -0.25:
        return "negative"
    elif score >= 0.25:
        return "positive"
    else:
        return "neutral"

df["sentiment_label"] = df["sentimentScorePortuguese"].apply(score_to_label)

# Verificação
print("Linhas finais:", len(df))
print(df["sentiment_label"].value_counts())

# Rodar o comparator
comp = ClassificationComparator(
    df=df,
    text_column="headlinePortuguese",
    label_column="sentiment_label"
)

resultados = comp.run_all()
print(resultados)

# preprocessing_variants.py

from preprocessing import (
    normalize_text,
    remove_stopwords,
    apply_stemming,
    apply_lemmatization
)

def preprocess_lemmatization(df, text_column="headlinePortuguese"):
    df = df.copy()

    df["text_clean"] = (
        df[text_column]
        .astype(str)
        .apply(normalize_text)
        .apply(remove_stopwords)
        .apply(apply_lemmatization)
    )
    return df


def preprocess_stemming(df, text_column="headlinePortuguese"):
    df = df.copy()

    df["text_clean"] = (
        df[text_column]
        .astype(str)
        .apply(normalize_text)
        .apply(remove_stopwords)
        .apply(apply_stemming)
    )
    return df

# ---
# Item C)
# ---

# Comparação: lemmatização vs stemming usando a melhor técnica de extração

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class LemmaStemComparison:

    def __init__(self, df, text_column, label_column, best_feature="TF-IDF"):
        self.df = df
        self.text_column = text_column
        self.label_column = label_column
        self.best_feature = best_feature
        self.results = []

    # Treinamento
    def evaluate(self, X, y, prep_name):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        clf = LogisticRegression(max_iter=300)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        self.results.append({
            "Pré-processamento": prep_name,
            "Acurácia": acc,
            "Feature Extraction": self.best_feature
        })

    # Executar comparação
    def run(self):

        y = self.df[self.label_column]

        # ----------------- LEMMATIZAÇÃO -----------------
        df_lemma = preprocess_lemmatization(self.df, self.text_column)
        fe_lemma = FeatureExtractor(df_lemma["text_clean"].tolist())

        if self.best_feature == "TF-IDF":
            X_lemma, _ = fe_lemma.tfidf()
        elif self.best_feature == "Bag of Words":
            X_lemma, _ = fe_lemma.bag_of_words()
        elif self.best_feature == "Word2Vec":
            X_lemma, _ = fe_lemma.word2vec()
        elif self.best_feature == "Coocorrência":
            X_lemma, _ = fe_lemma.cooccurrence_matrix()
        else:
            X_lemma = fe_lemma.statistical_analysis()

        self.evaluate(X_lemma, y, "Lemmatização")


        # ----------------- STEMMING -----------------
        df_stem = preprocess_stemming(self.df, self.text_column)
        fe_stem = FeatureExtractor(df_stem["text_clean"].tolist())

        if self.best_feature == "TF-IDF":
            X_stem, _ = fe_stem.tfidf()
        elif self.best_feature == "Bag of Words":
            X_stem, _ = fe_stem.bag_of_words()
        elif self.best_feature == "Word2Vec":
            X_stem, _ = fe_stem.word2vec()
        elif self.best_feature == "Coocorrência":
            X_stem, _ = fe_stem.cooccurrence_matrix()
        else:
            X_stem = fe_stem.statistical_analysis()

        self.evaluate(X_stem, y, "Stemming")

        # ----------------- Resultado final -----------------
        return pd.DataFrame(self.results)