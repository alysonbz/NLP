import pandas as pd

# 1) Um função que retorne a quantidade de sentenças em um texto.
def count_sentences(text: str):
    return len([s for s in text.split('.') if s.strip()])

# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
def count_capitalized_words(text: str):
    return len([w for w in text.split() if w and w[0].isalpha() and w[0].isupper()])

# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.
def count_numeric_characters(text: str):
    return len([char for char in text if char.isdigit()])

# 4) Uma função que retorne a quantidade de palavras que estão  em caixa alta.
def count_uppercase_words(text: str):
    return len([word for word in text.split() if word.isupper()])

# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.
texts = [
    "Hoje é um bom dia. Estou feliz!",
    "O aluno tirou nota 10 em Matemática!",
    "Este TEXTO TEM PALAVRAS EM CAIXA ALTA.",
    "Python é incrível, versão 3.11 já estável."
]
df = pd.DataFrame({"texto": texts})


# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.
df_results = pd.DataFrame({
    "sentencas": df["texto"].apply(count_sentences),
    "palavras_maiusc": df["texto"].apply(count_capitalized_words),
    "numericos": df["texto"].apply(count_numeric_characters),
    "caixa_alta": df["texto"].apply(count_uppercase_words),
})

print(df_results)