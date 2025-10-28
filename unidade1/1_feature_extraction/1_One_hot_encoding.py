import pandas as pd
from src.utils import load_df1_one_hot

df1 = load_df1_one_hot()

# Print the features of df1
print(df1.columns.tolist())
print("\n")

# Perform one-hot encoding on column "feature 5"
df1 = pd.get_dummies(df1, columns=['feature 5'])

# Print the new features of df1
print(df1.columns.tolist())
print("\n")

# Print first five rows of df1
print("Primeiras 5 linhas do DataFrame após One Hot Encoding:")
print(df1.head())

# Implementação manual do Count Vectorizer

# Exemplo de corpus
corpus = [
    "o gato está no telhado",
    "o cachorro está no jardim",
    "o gato e o cachorro estão dormindo"
]

# Tokenização simples e criação do vocabulário
vocab = set()
for doc in corpus:
    tokens = doc.lower().split()
    vocab.update(tokens)

# Ordena o vocabulário para consistência
vocab = sorted(vocab)
print("Vocabulário:", vocab)

# Cria uma matriz de contagem
count_matrix = []
for doc in corpus:
    tokens = doc.lower().split()
    counts = [tokens.count(word) for word in vocab]
    count_matrix.append(counts)

# Converte em DataFrame para visualização
import pandas as pd
df = pd.DataFrame(count_matrix, columns=vocab)
print("\nMatriz de Contagem (Count Vectorizer manual):")
print(df)