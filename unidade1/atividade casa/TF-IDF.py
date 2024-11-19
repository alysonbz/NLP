import pandas as pd

def load_dataset_sentences():
    columns = ["S.No.", "Sentences"]
    data = [[1, "I like watch movies"], [2, "I dont want go to they"], [3, "Iám fine and you?"]]
    return pd.DataFrame(data, columns=columns)

def count_sentences(sentence):
    palavra = sentence.split()
    return len(palavra)

df = load_dataset_sentences()
df['contagem_de_palavra'] = df['Sentences'].apply(count_sentences)

total_words = df['contagem_de_palavra'].sum()
df['contagem_dividida'] = df['contagem_de_palavra'].nunique() / total_words

print(df.head())
