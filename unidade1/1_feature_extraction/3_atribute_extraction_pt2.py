from src.utils import load_pfizer_tweet_dataset
import matplotlib.pyplot as plt

# Carregar o conjunto de dados de tweets do Pfizer
tweets = load_pfizer_tweet_dataset()


## WORD COUNT

# a) Function that returns number of words in a string
def count_words(string):
    # Split the string into words
    words = string.split()

    # Return the number of words
    return len(words)


# Create a new feature word_count
tweets['word_count'] = tweets['text'].apply(count_words)

# Print the average word count of the tweets
print("Word count mean: ", tweets['word_count'].mean())

## CHAR COUNT

# Create a feature char_count from "text" feature. Use len function to count
tweets['char_count'] = tweets['text'].apply(len)

# Print the average character count
print("Char count mean: ", tweets['char_count'].mean())


## HASHTAGS COUNT

# Function that returns number of hashtags in a string
def count_hashtags(string):
    # Split the string into words
    words = string.split()

    # Create a list of words that are hashtags
    hashtags = [word for word in words if word.startswith('#')]

    # Return number of hashtags
    return len(hashtags)


# Create a feature hashtag_count and display distribution
tweets['hashtag_count'] = tweets['text'].apply(count_hashtags)
tweets['hashtag_count'].hist()
plt.title('Hashtag count distribution')
plt.xlabel('Number of Hashtags')
plt.ylabel('Frequency')
plt.show()


## MENTIONS COUNT

# Function that returns number of mentions in a string
def count_mentions(string):
    # Split the string into words
    words = string.split()

    # Create a list of words that are mentions
    mentions = [word for word in words if word.startswith('@')]

    # Return number of mentions
    return len(mentions)


# Create a feature mention_count and display distribution
tweets['mention_count'] = tweets['text'].apply(count_mentions)
tweets['mention_count'].hist()
plt.title('Mention count distribution')
plt.xlabel('Number of Mentions')
plt.ylabel('Frequency')
plt.show()
