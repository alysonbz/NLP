# Atividade
# Um função que retorne a quantidade de sentenças em um texto.
def contar_sentencas(texto):
    # Conta o número de ocorrências do caractere '.' no texto
    return texto.count('.')

# Exemplo de uso:
texto_exemplo = "Em uma manhã ensolarada, Maria acordou cedo e preparou um café da manhã nutritivo. Ela abriu a janela e sentiu a brisa fresca enquanto observava os pássaros voando no céu. Depois, saiu para caminhar no parque, onde encontrou seu amigo Pedro. Eles conversaram sobre os planos para o fim de semana e riram muito."


quantidade_sentencas = contar_sentencas(texto_exemplo)
print(f"Quantidade de sentenças: {quantidade_sentencas}")

# Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.
def contar_palavras_maiusculas(texto):
    palavras = texto.split()
    contagem = sum(1 for palavra in palavras if palavra[0].isupper())
    return contagem
#exemplo de uso
texto = "O Flamengo é o melhor time do Brasil e do mundo!"
resultado = contar_palavras_maiusculas(texto)
print(f"Quantidade de palavras que iniciam com letra maiuscula: {resultado}")
# Uma função que retorne a quantidade de caracteres numéricos em um texto.
def contar_numeros(texto):
    contagem = sum(1 for caractere in texto if caractere.isdigit())
    return contagem

#exemplo
texto = "No ano de 2019 o CRFlamengo conquistou a triplice da cora conquistando 3 titulos em uma unica temporada!"
resultado = contar_numeros(texto)
print(f"A quantidade de caracteres numericos e: {resultado}")
# Uma função que retorne a quantidade de palavras que estão  em caixa alta.
def contar_palavras_caixa_alta(texto):
    palavras = texto.split()
    contagem = sum(1 for palavra in palavras if palavra.isupper())
    return contagem

texto = "Todos os times a seguir em caixa alta são os melhores do mundo: FLAMENGO, CEARA, vasco, corinthians, palmeiras, fortaleza"
resultado = contar_palavras_caixa_alta(texto)
print(f"A quantidade de palavras em caixa alta na frase é: {resultado}")

#Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.
import pandas as pd

# Textos de exemplo
textos = [
    "Em uma manhã ensolarada, Maria acordou cedo e preparou um café da manhã nutritivo. Ela abriu a janela e sentiu a brisa fresca enquanto observava os pássaros voando no céu. Depois, saiu para caminhar no parque, onde encontrou seu amigo Pedro. Eles conversaram sobre os planos para o fim de semana e riram muito.",
    "O Flamengo é o melhor time do Brasil e do mundo!",
    "No ano de 2019 o CRFlamengo conquistou a triplice da cora conquistando 3 titulos em uma unica temporada!",
    "Todos os times a seguir em caixa alta são os melhores do mundo: FLAMENGO, CEARA, vasco, corinthians, palmeiras, fortaleza"
]
df_textos = pd.DataFrame({'Textos': textos})
print(df_textos)

#Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.  Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5

df_resultados = pd.DataFrame({
    'Palavras maiusculas': df_textos['Textos'].apply(contar_palavras_maiusculas),
    'Numeros': df_textos['Textos'].apply(contar_numeros),
    'Palavras em caixa alta': df_textos['Textos'].apply(contar_palavras_caixa_alta),
    'Sentenças': df_textos['Textos'].apply(contar_sentencas)
})
print(df_resultados)
