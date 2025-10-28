import pandas as pd
from src.utils import load_movie_review_clean_dataset
import re

df = load_movie_review_clean_dataset()

def count_vectorizer(list_to_vect):
    #tokenizar - pode ser com biblioteca
    tokenized_docs = []
    for text in list_to_vect:
        tokens = re.findall(r'\b\w+\b', text.lower())
        tokenized_docs.append(tokens)

    #criar saco de palavra (id para cada termo)
    vocab = sorted(set(word for tokens in tokenized_docs for word in tokens))

    #criar a matriz de dicionario para cada sentença
    count_data = []
    for tokens in tokenized_docs:
        word_counts = {word: tokens.count(word) for word in vocab}
        count_data.append(word_counts)

    # converte para DataFrame
    df_count = pd.DataFrame(count_data, columns=vocab)

    return df_count

# Print dos textos originais
print("Textos originais:")
print(df['review'], "\n")

# Executa o Count Vectorizer manual
print("Matriz de contagem (Count Vectorizer manual):")
print(count_vectorizer(df['review']))
