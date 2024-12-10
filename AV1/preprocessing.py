import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from string import punctuation
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Baixar os recursos necessários para o NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# Configurar stopwords para português
stopwords_pt = set(stopwords.words('portuguese'))

# Função para limpar o texto
def clean_text(text):
    text = text.lower()  # Caixa baixa
    text = re.sub(r'@\S+', '', text)  # Remoção de menções
    text = re.sub(r'<[^<]+?>', '', text)  # Remoção de tags HTML
    text = ''.join(c for c in text if not c.isdigit())  # Remoção de números
    text = re.sub(r'(www\.\S+|https?://\S+)', '', text)  # Remoção de URLs
    text = ''.join(c for c in text if c not in re.escape(punctuation))  # Remoção de pontuações
    return text

# Função para tokenização e remoção de stopwords
def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text, language="portuguese")
    return [word for word in tokens if word not in stopwords_pt]

# Função para aplicar stemming
def apply_stemming(tokens):
    stemmer = RSLPStemmer()
    return [stemmer.stem(token) for token in tokens]

# Função principal de pré-processamento
def preprocess_text(text):
    clean = clean_text(text)
    tokens = tokenize_and_remove_stopwords(clean)
    stemmed_tokens = apply_stemming(tokens)
    return " ".join(stemmed_tokens)

# Função para processar o dataset completo
def preprocess_dataset(df, text_col):
    df["clean_text"] = df[text_col].apply(clean_text)
    df["processed_text"] = df[text_col].apply(preprocess_text)
    return df[["id", "clean_text", "processed_text"]]

# Classe FeatureExtractor
class FeatureExtractor:
    def __init__(self, corpus):
        self.corpus = corpus

    def statistical_analysis(self):
        word_counts = [len(text.split()) for text in self.corpus]
        stats = {
            "mean_length": np.mean(word_counts),
            "max_length": np.max(word_counts),
            "min_length": np.min(word_counts),
            "total_tokens": np.sum(word_counts),
            "var_tokens": np.var(word_counts)
        }
        return stats

    def count_vectorizer(self):
        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(self.corpus)
        return count_matrix, vectorizer.get_feature_names_out()

    def tfidf_vectorizer(self):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.corpus)
        return tfidf_matrix, vectorizer.get_feature_names_out()

    def cooccurrence_matrix(self, window_size=2):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.corpus)
        vocab = vectorizer.get_feature_names_out()
        cooccurrence = (X.T @ X).toarray()
        np.fill_diagonal(cooccurrence, 0)  # Remover autocorrelações
        return pd.DataFrame(cooccurrence, index=vocab, columns=vocab)

    def word2vec(self, vector_size=100, window=5, min_count=1, epochs=10):
        tokenized_corpus = [text.split() for text in self.corpus]
        model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count
        )
        model.train(tokenized_corpus, total_examples=len(tokenized_corpus), epochs=epochs)
        return model

# Carregar o dataset
file_path = "dataset_final.csv"  # Atualize o caminho se necessário
df = pd.read_csv(file_path)

# Ajuste na coluna 'thematic_coherence'
df['thematic_coherence'] = df['thematic_coherence'] - 1

# Processar o dataset
processed_df = preprocess_dataset(df, text_col="essay")

# Criar uma instância da classe FeatureExtractor
corpus = processed_df["processed_text"].values
extractor = FeatureExtractor(corpus)

# Usar os métodos da classe
stats = extractor.statistical_analysis()
count_matrix, count_features = extractor.count_vectorizer()
tfidf_matrix, tfidf_features = extractor.tfidf_vectorizer()
cooc_matrix = extractor.cooccurrence_matrix()
word2vec_model = extractor.word2vec()

# Exibir os resultados
print("Estatísticas:", stats)
print("Count Vectorizer - Features:", count_features[:10])  # Exemplo com 10 primeiros
print("TF-IDF - Features:", tfidf_features[:10])  # Exemplo com 10 primeiros
print("Co-occurrence Matrix:\n", cooc_matrix.head())  # Mostra as primeiras linhas
print("Word2Vec Model:", word2vec_model)

# Salvar o dataset processado
processed_df.to_csv("processed_dataset.csv", index=False)

# Exibir as primeiras linhas do dataset processado
print(processed_df.head())
