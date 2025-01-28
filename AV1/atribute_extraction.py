import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# Certifique-se de baixar as stopwords do NLTK, caso ainda não tenha feito
from gensim.models import Word2Vec

# Exemplo de matriz TF-IDF
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Certifique-se de baixar as stopwords do NLTK, caso ainda não tenha feito
import nltk
nltk.download('stopwords')
df = pd.read_csv('article_nlp_datasets.csv')
stop_words = stopwords.words('portuguese')
print(stop_words)
print(df['premise_processed'].head())
print(df['hypothesis_processed'].head())

df['premise_processed'] = df['premise_processed'].fillna('')
df['hypothesis_processed'] = df['hypothesis_processed'].fillna('')


class AttributeExtractor:
    def __init__(self, tokenized_premises, tokenized_hypotheses):
        self.tokenized_premises = tokenized_premises
        self.tokenized_hypotheses = tokenized_hypotheses

    def individual_statistical_analysis(self):
        """
        Calculates basic statistics like word count and unique word count for premises and hypotheses.
        """
        stats = []
        for premise, hypothesis in zip(self.tokenized_premises, self.tokenized_hypotheses):
            stats.append({
                'premise_word_count': len(premise),
                'hypothesis_word_count': len(hypothesis),
                'premise_unique_words': len(set(premise)),
                'hypothesis_unique_words': len(set(hypothesis))
            })
        return pd.DataFrame(stats)

    def count_vectorizer_features(self):
        """
        Extracts features using CountVectorizer.
        """
        #combined_texts = [" ".join(tokens) for tokens in self.tokenized_premises + self.tokenized_hypotheses]
        #print(combined_texts[:5])
        combined_texts = df['premise_processed'].astype(str) + " " + df['hypothesis_processed'].astype(str)
        print("Combined texts (first 5):", combined_texts[:2])

        # Configurando o CountVectorizer com stopwords em português
        vectorizer = CountVectorizer(stop_words=stop_words, token_pattern=r'\b[a-zA-Z]{3,}\b', min_df=2)

        count_matrix = vectorizer.fit_transform(combined_texts)
        return count_matrix, vectorizer.get_feature_names_out()

    def tfidf_features(self):
        """
        Extracts features using TF-IDF Vectorizer.
        """

        combined_texts = [" p".join(tokens) for tokens in df['premise_processed'] + df['hypothesis_processed']]
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts)
        return tfidf_matrix, tfidf_vectorizer.get_feature_names_out()

    def co_occurrence_matrix(self):
        """
        Constructs a co-occurrence matrix for premises and hypotheses.
        """
        all_tokens = [word for tokens in self.tokenized_premises + self.tokenized_hypotheses for word in tokens]
        vocab = list(set(all_tokens))
        vocab_index = {word: i for i, word in enumerate(vocab)}

        co_matrix = np.zeros((len(vocab), len(vocab)))

        for tokens in self.tokenized_premises + self.tokenized_hypotheses:
            for i, word in enumerate(tokens):
                if word in vocab_index:
                    word_idx = vocab_index[word]
                    for j in range(max(0, i - 2), min(len(tokens), i + 3)):
                        if tokens[j] in vocab_index:
                            co_matrix[word_idx][vocab_index[tokens[j]]] += 1

        return pd.DataFrame(co_matrix, index=vocab, columns=vocab)

    def word2vec_features(self, vector_size=100, window=5, min_count=1):
        """
        Trains a Word2Vec model and returns the vector representations for each word.
        """
        model = Word2Vec(
            sentences=self.tokenized_premises + self.tokenized_hypotheses,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        word_vectors = {word: model.wv[word] for word in model.wv.index_to_key}
        return word_vectors


# Usage example
if __name__ == "__main__":
    # Assuming `df` contains preprocessed `premise_processed` and `hypothesis_processed`
    extractor = AttributeExtractor(
        tokenized_premises=df['premise_processed'],
        tokenized_hypotheses=df['hypothesis_processed']
    )

    # Individual Statistical Analysis
    stats_df = extractor.individual_statistical_analysis()

    plt.figure(figsize=(10, 6))
    plt.hist(stats_df['premise_word_count'], bins=30, alpha=0.7, label='Premise')
    plt.hist(stats_df['hypothesis_word_count'], bins=30, alpha=0.7, label='Hypothesis')
    plt.xlabel('Número de Palavras')
    plt.ylabel('Frequência')
    plt.title('Distribuição do Número de Palavras')
    plt.legend()
    plt.show()
    print("Statistical Analysis:")
    print(stats_df.head())

    # CountVectorizer Features
    count_matrix, count_features = extractor.count_vectorizer_features()

    word_counts = count_matrix.sum(axis=0).A1  # Soma total das palavras na matriz
    word_freq_df = pd.DataFrame({'word': count_features, 'count': word_counts})
    top_words = word_freq_df.sort_values(by='count', ascending=False).head(20)

    # Gráfico de barras
    plt.figure(figsize=(12, 6))
    sns.barplot(x='count', y='word', data=top_words, palette='viridis')
    plt.xlabel('Frequência')
    plt.ylabel('Palavras')
    plt.title('Palavras Mais Frequentes (CountVectorizer)')
    plt.show()

    print("Count Vectorizer Features:")
    print(count_matrix.toarray())
    print(count_features)

    # TF-IDF Features
    tfidf_matrix, tfidf_features = extractor.tfidf_features()
    print("TF-IDF Features:")
    print(tfidf_matrix.toarray())
    print(tfidf_features)

    # Co-occurrence Matrix
    co_matrix = extractor.co_occurrence_matrix()
    print("Co-occurrence Matrix:")
    print(co_matrix.head())

    # Word2Vec Features
    word_vectors = extractor.word2vec_features()
    print("Word2Vec Features:")
    print(word_vectors)

    tfidf_dense_matrix = tfidf_matrix.toarray()  # Converte para matriz densa
    # Visualizar apenas as primeiras 10x10 entradas da matriz TF-IDF
    sns.heatmap(tfidf_dense_matrix[:10, :10], annot=True, fmt=".2f")
    plt.title("Exemplo de matriz TF-IDF")
    plt.show()

    # Histograma para número de palavras


