import pandas as pd
from src.utils import load_movie_review_clean_dataset

df = load_movie_review_clean_dataset()


def count_vectorizer(list_to_vect):
    #tokenizar
    #criar saco de palavra (id para cada termo)
    #criar a matriz de dicionario para cada sentença
    #seu código aqui
    return None

print(df['review'].head(1))
print(count_vectorizer(df['review']))
