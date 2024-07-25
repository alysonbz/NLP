import pandas as pd

# Define the functions
def count_sentences(text):
    sentences = text.split('.')
    return len(sentences)

def count_capitalized_words(text):
    words = text.split()
    count = 0
    for word in words:
        if word[0].isupper():
            count += 1
    return count

def count_numeric_characters(text):
    count = 0
    for char in text:
        if char.isdigit():
            count += 1
    return count

def count_uppercase_words(text):
    words = text.split()
    count = 0
    for word in words:
        if word.isupper():
            count += 1
    return count

# Example texts
texts = [
    "Hello world. This is a sample text.",
    "It contains 3 sentences. There are 5 capitalized words.",
    "The number of numeric characters is 12345.",
    "There are 2 uppercase words. AAAAAAH"
]

# Create a dataframe
df = pd.DataFrame(texts, columns=['Text'])

# Apply the functions to create new columns
df['Number of sentences'] = df['Text'].apply(count_sentences)
df['Number of capitalized words'] = df['Text'].apply(count_capitalized_words)
df['Number of numeric characters'] = df['Text'].apply(count_numeric_characters)
df['Number of uppercase words'] = df['Text'].apply(count_uppercase_words)

# Print the dataframe
print(df)   