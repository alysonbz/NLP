from src.utils import load_pfizer_tweet_dataset
import matplotlib.pyplot as plt

# Carrega o dataset de tweets
tweets = load_pfizer_tweet_dataset()

## WORD COUNT

# a) Função que retorna o número de palavras em uma string
def count_words(string):
    # Divide a string em palavras
    words = string.split()

    # Retorna o número de palavras
    return len(words)

# Cria uma nova coluna 'word_count' no DataFrame
tweets['word_count'] = tweets['text'].apply(count_words)

# Imprime a média de palavras dos tweets
print("Word count mean: ", tweets['word_count'].mean())

## CHAR COUNT

# Cria uma coluna 'char_count' contando o número de caracteres usando a função len
tweets['char_count'] = tweets['text'].apply(len)

# Imprime a média de caracteres dos tweets
print("Char count mean: ", tweets['char_count'].mean())

## HASHTAGS COUNT

# Função que retorna o número de hashtags em uma string
def count_hashtags(string):
    # Divide a string em palavras
    words = string.split()

    # Cria uma lista de palavras que são hashtags
    hashtags = [word for word in words if word.startswith('#')]

    # Retorna o número de hashtags
    return len(hashtags)

# Cria uma coluna 'hashtag_count' e exibe a distribuição
tweets['hashtag_count'] = tweets['text'].apply(count_hashtags)
tweets['hashtag_count'].hist()
plt.title('Hashtag count distribution')
plt.show()

## MENTIONS COUNT

# Função que retorna o número de menções em uma string
def count_mentions(string):
    # Divide a string em palavras
    words = string.split()

    # Cria uma lista de palavras que são menções
    mentions = [word for word in words if word.startswith('@')]

    # Retorna o número de menções
    return len(mentions)

# Cria uma coluna 'mention_count' e exibe a distribuição
tweets['mention_count'] = tweets['text'].apply(count_mentions)
tweets['mention_count'].hist()
plt.title('Mention count distribution')
plt.show()
