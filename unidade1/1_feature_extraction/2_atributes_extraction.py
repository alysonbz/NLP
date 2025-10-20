import pandas as pd
from src.utils import load_movie_review_clean_dataset


# Atividade
# 1) Um função que retorne a quantidade de sentenças em um texto.
def count_sentences(text: str):
    return len(text.split(".")) # Considerando que as sentenças terminam com ponto final.

# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
def count_capitalized_words(text: str):
    return len([word for word in text.split() if word[0] == word[0].upper()])

# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.
def count_numeric_characters(text: str):
    return len([char for char in text if char.isdigit()])

# 4) Uma função que retorne a quantidade de palavras que estão  em caixa alta.
def count_uppercase_words(text: str):
    return len([word for word in text.split() if word.isupper()])

# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.
df = load_movie_review_clean_dataset()
# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.
df['count_sentences_review'] = df['review'].apply(count_sentences)
df['count_capitalized_review'] = df['review'].apply(count_capitalized_words)
df['count_numeric_review'] = df['review'].apply(count_numeric_characters)
df['count_uppercase_review'] = df['review'].apply(count_uppercase_words)

print(df.head())