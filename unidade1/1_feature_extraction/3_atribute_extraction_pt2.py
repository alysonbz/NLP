from src.utils import load_pfizer_tweet_dataset
import matplotlib.pyplot as plt

# Carregar o conjunto de dados de tweets do Pfizer
tweets = load_pfizer_tweet_dataset()

## CONTAGEM DE PALAVRAS

# a) Função que retorna o número de palavras em uma string
def count_words(string):
    # Dividir a string em palavras
    words = string.split()

    # Retornar o número de palavras
    return len(words)

# Criar uma nova coluna 'word_count'
tweets['word_count'] = tweets['text'].apply(count_words)

# Imprimir a média da contagem de palavras dos tweets
print("Média da contagem de palavras: ", tweets['word_count'].mean())

## CONTAGEM DE CARACTERES

# Criar a coluna 'char_count' contando os caracteres da coluna "text"
tweets['char_count'] = tweets['text'].apply(len)

# Imprimir a média da contagem de caracteres
print("Média da contagem de caracteres: ", tweets['char_count'].mean())

## CONTAGEM DE HASHTAGS

# Função que retorna o número de hashtags em uma string
def count_hashtags(string):
    # Dividir a string em palavras
    words = string.split()

    # Criar uma lista de palavras que são hashtags
    hashtags = [word for word in words if word.startswith('#')]

    # Retornar o número de hashtags
    return len(hashtags)

# Criar a coluna 'hashtag_count' e exibir a distribuição
tweets['hashtag_count'] = tweets['text'].apply(count_hashtags)
tweets['hashtag_count'].hist()
plt.title('Distribuição da contagem de hashtags')
plt.xlabel('Número de hashtags')
plt.ylabel('Frequência')
plt.show()

## CONTAGEM DE MENÇÕES

# Função que retorna o número de menções em uma string
def count_mentions(string):
    # Dividir a string em palavras
    words = string.split()

    # Criar uma lista de palavras que são menções
    mentions = [word for word in words if word.startswith('@')]

    # Retornar o número de menções
    return len(mentions)

# Criar a coluna 'mention_count' e exibir a distribuição
tweets['mention_count'] = tweets['text'].apply(count_mentions)
tweets['mention_count'].hist()
plt.title('Distribuição da contagem de menções')
plt.xlabel('Número de menções')
plt.ylabel('Frequência')
plt.show()
