import pandas as pd
from src.utils import load_movie_review_dataset
from sklearn.feature_extraction.text import CountVectorizer

# Carregar o conjunto de dados
corpus = load_movie_review_dataset()
corpus = corpus['tagline'].dropna()

# Criar objeto CountVectorizer
vectorizer = CountVectorizer()

# Gerar a matriz de vetores de palavras
bow_matrix = vectorizer.fit_transform(corpus)

# Converter bow_matrix em um DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Mapear os nomes das colunas para o vocabulário
bow_df.columns = vectorizer.get_feature_names_out()

# Imprimir bow_df
print(bow_df)
