import pandas as pd
from src.utils import load_movie_review_clean_dataset
import re

df = load_movie_review_clean_dataset()


def count_vectorizer(list_to_vect):
    # Tokenização - letras minúsculas + remover pontuação
    tokenized_docs = [
        re.findall(r'\b\w+\b', doc.lower()) for doc in list_to_vect
    ]
    
    # Vocabulário (todas as palavras únicas)
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    
    # Matriz de contagem
    matrix = []
    for doc in tokenized_docs:
        vector = [0] * len(vocab)
        for word in doc:
            if word in vocab_index:
                vector[vocab_index[word]] += 1
        matrix.append(vector)
    
    # DataFrame para visualização mais fácil
    count_df = pd.DataFrame(matrix, columns=vocab)
    return count_df

print(df['review'].head())
print(count_vectorizer(df['review'].head()))
