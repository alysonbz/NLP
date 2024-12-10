import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Baixar pacotes necessários do NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


class Preprocessor:
    """
    Classe para realizar o pré-processamento de texto.
    """

    def __init__(self, language='portuguese'):
        """
        Inicializa a classe com idioma para stopwords e stemmer.
        """
        self.language = language
        try:
            self.stopwords = set(stopwords.words(language))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)

    @staticmethod
    def remove_punctuation(text):
        """Remove pontuação de um texto."""
        return text.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def to_lowercase(text):
        """Converte texto para letras minúsculas."""
        return text.lower()

    @staticmethod
    def remove_mentions(text):
        """Remove menções (@usuario) de um texto."""
        return re.sub(r'@\w+', '', text)

    def remove_stopwords(self, text):
        """Remove stopwords do texto."""
        tokens = word_tokenize(text)
        filtered_words = [word for word in tokens if word not in self.stopwords]
        return ' '.join(filtered_words)

    def apply_stemming(self, text):
        """Aplica stemming ao texto."""
        tokens = word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed_words)

    def preprocess_text(self, text):
        """
        Aplica todas as etapas de pré-processamento em um texto.
        """
        if not isinstance(text, str) or not text.strip():
            return ''
        text = self.remove_mentions(text)
        text = self.to_lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)
        text = self.apply_stemming(text)
        return text

    def preprocess_dataframe(self, df, text_column):
        """
        Aplica o pré-processamento em uma coluna específica de um DataFrame.
        """
        if text_column not in df.columns:
            raise ValueError(f"A coluna '{text_column}' não existe no DataFrame.")

        df = df.copy()
        df[text_column] = df[text_column].apply(self.preprocess_text)
        return df


if __name__ == "__main__":
    # Exemplo de uso
    dataset_path = r"C:\Users\MASTER\OneDrive\Área de Trabalho\NLP Aleky\NLP\AV1\brazilian_headlines_sentiments.csv"
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado no caminho especificado: {dataset_path}")

    preprocessor = Preprocessor()
    df_preprocessed = preprocessor.preprocess_dataframe(df, 'headlinePortuguese')

    # Salva o resultado para futura análise
    output_path = dataset_path.replace('.csv', '_preprocessed.csv')
    df_preprocessed.to_csv(output_path, index=False)
    print(f"Pré-processamento concluído. Arquivo salvo em: {output_path}")
