from ..src.utils import load_pfizer_tweet_dataset

tweets = load_pfizer_tweet_dataset()
import matplotlib.pyplot as plt

## WORD COUNT

# a) Function that returns number of words in a string
def count_words(string):
    # Split the string into words
    words = split.string

    # Return the number of words
    return len(words)

print(tweets.head())

# Create a new feature word_count
tweets['word_count'] = tweets[1].apply(count_words)
