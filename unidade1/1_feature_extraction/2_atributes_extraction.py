# Atividade

import pandas as pd
from src.utils import load_movie_review_dataset

# 1) Um função que retorne a quantidade de sentenças em um texto.

def count_sentences(text):
    return sum(1 for char in text if char in '.!?')

# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.

def count_capitalized_words(text):
    words = text.split()
    return sum(1 for word in words if word[0].isupper())

# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.

def count_numeric_chars(text):
    return sum(c.isdigit() for c in text)

# 4) Uma função que retorne a quantidade de palavras que estão em caixa alta.

def count_uppercase_words(text):
    words = text.split()
    return sum(1 for word in words if word.isupper())

# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.
texts = [
    "Hoje é um belo dia. Vamos aproveitar o sol!",
    "A NASA descobriu uma nova estrela a 3000 anos-luz.",
    "EU GOSTO DE PROGRAMAR EM PYTHON!",
    "O número 2025 será um grande ano para a tecnologia."
]
df_texts = pd.DataFrame({'texto': texts})


# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.

df_results = pd.DataFrame({
    'sentencas': df_texts['texto'].apply(count_sentences),
    'palavras_iniciais_maiusculas': df_texts['texto'].apply(count_capitalized_words),
    'caracteres_numericos': df_texts['texto'].apply(count_numeric_chars),
    'palavras_caixa_alta': df_texts['texto'].apply(count_uppercase_words)
})

print("criação:", df_results)


#test

df_movie = load_movie_review_dataset()

df_texts1 = df_movie[["overview"]].head(4)

df_results1 = pd.DataFrame({
    'sentencas': df_texts1["overview"].apply(count_sentences),
    'palavras_iniciais_maiusculas': df_texts1["overview"].apply(count_capitalized_words),
    'caracteres_numericos': df_texts1["overview"].apply(count_numeric_chars),
    'palavras_caixa_alta': df_texts1["overview"].apply(count_uppercase_words)
})

print("dataset:", df_results1)