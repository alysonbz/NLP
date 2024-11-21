import pandas as pd

# Atividade
# 1) Um função que retorne a quantidade de sentenças em um texto.
texts = 'A pessoa Errou muito o Numero de palavras Certas, 9 8 4 3 21, e a OUTRA acertou todas as palavras CERTAS 5 4 8 3 1'
def word_count(text):
    words = text.split()
    return len(words)

print(word_count(texts))



# 2) Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.

def word_countmaius(text):
    words = text.split()
    return len([word for word in words if word[0].isupper()])

print(word_countmaius(texts))



# 3) Uma função que retorne a quantidade de caracteres numéricos em um texto.
def num_counts(text):
    nume = text.split()
    return len([1 for word in nume if word.isdigit()])


print(num_counts(texts))

# 4) Uma função que retorne a quantidade de palavras que estão  em caixa alta.
def word_countupper(text):
    words = text.split()
    return len([word for word in words if word.isupper()])

print(word_countupper(texts))


# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.



data = {
    "Texto": [
        "A pessoa Errou muito o Numero de palavras Certas.",
        "A pessoa acertou todas as palavras CERTAS.",
        "9 8 4 3 21.",
        "OUTRA acertou todas as palavras CERTAS."
    ]
}
df = pd.DataFrame(data)

print("\nDataFrame:")
print(df)

# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.

# Aplicando as funções e adicionando colunas ao DataFrame
df["Quantidade de Palavras"] = df["Texto"].apply(word_count)
df["Palavras com Primeira Maiúscula"] = df["Texto"].apply(word_countmaius)
df["Números no Texto"] = df["Texto"].apply(num_counts)
df["Palavras em Caixa Alta"] = df["Texto"].apply(word_countupper)

# Exibindo o DataFrame final
print("\nDataFrame Gerado:")
print(df)