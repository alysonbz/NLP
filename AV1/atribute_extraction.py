import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import coo_matrix
from gensim.models import Word2Vec
from typing import List, Tuple, Dict


class AttributeExtractor:

    def __init__(self, text_column: str):
        self.text_column = text_column
        self.vectorizer = None
        self.word2vec_model = None

    def analyze_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        if self.text_column not in df.columns:
            raise ValueError(f"Coluna '{self.text_column}' não encontrada no DataFrame.")

        df['word_count'] = df[self.text_column].apply(lambda x: len(str(x).split()))
        df['char_count'] = df[self.text_column].apply(lambda x: len(str(x)))
        stats = {
            'avg_word_count': df['word_count'].mean(),
            'avg_char_count': df['char_count'].mean(),
            'max_word_count': df['word_count'].max(),
            'max_char_count': df['char_count'].max(),
        }
        return stats

    def extract_count_vectorizer(self, df: pd.DataFrame, max_features: int = 5000) -> Tuple[coo_matrix, List[str]]:
        self.vectorizer = CountVectorizer(max_features=max_features)
        count_matrix = self.vectorizer.fit_transform(df[self.text_column])
        return count_matrix, self.vectorizer.get_feature_names_out().tolist()

    def extract_tfidf(self, df: pd.DataFrame, max_features: int = 5000) -> Tuple[coo_matrix, List[str]]:
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = self.vectorizer.fit_transform(df[self.text_column])
        return tfidf_matrix, self.vectorizer.get_feature_names_out().tolist()

    def extract_cooccurrence_matrix(self, df: pd.DataFrame) -> Tuple[coo_matrix, List[str]]:
        self.vectorizer = CountVectorizer()
        count_matrix = self.vectorizer.fit_transform(df[self.text_column])
        cooccurrence = count_matrix.T.dot(count_matrix)
        return coo_matrix(cooccurrence), self.vectorizer.get_feature_names_out().tolist()

    def train_word2vec(self, df: pd.DataFrame, vector_size: int = 100, window: int = 5, min_count: int = 2) -> Word2Vec:
        sentences = [str(text).split() for text in df[self.text_column]]
        self.word2vec_model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
        return self.word2vec_model

    def get_word_embedding(self, word: str) -> List[float]:
        if self.word2vec_model is None:
            raise ValueError("Modelo Word2Vec ainda não foi treinado.")
        if word not in self.word2vec_model.wv:
            raise ValueError(f"A palavra '{word}' não está no vocabulário do modelo.")
        return self.word2vec_model.wv[word].tolist()


if __name__ == "__main__":
    dataset_path = r"C:\Users\MASTER\OneDrive\Área de Trabalho\NLP Aleky\NLP\AV1\brazilian_headlines_sentiments_preprocessed.csv"
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {dataset_path}")

    extractor = AttributeExtractor(text_column='headlinePortuguese')

    # 1. Análise Estatística
    stats = extractor.analyze_statistics(df)
    print("Estatísticas dos textos:", stats)

    # 2. CountVectorizer
    count_matrix, count_features = extractor.extract_count_vectorizer(df)
    print("Features (CountVectorizer):", count_features[:10])

    # 3. TF-IDF
    tfidf_matrix, tfidf_features = extractor.extract_tfidf(df)
    print("Features (TF-IDF):", tfidf_features[:10])

    # 4. Matriz de Coocorrência
    cooccurrence_matrix, cooccurrence_features = extractor.extract_cooccurrence_matrix(df)
    print("Features (Coocorrência):", cooccurrence_features[:10])

    # 5. Word2Vec
    word2vec_model = extractor.train_word2vec(df)
    print("Palavras no vocabulário:", list(word2vec_model.wv.index_to_key)[:10])
    print("Vetor de exemplo:", extractor.get_word_embedding('brasil'))
