import pandas as pd
from src.utils import load_df1_one_hot

# Carregar o dataframe df1 usando a função load_df1_one_hot
df1 = load_df1_one_hot()

# Print as características originais do dataframe df1
print("Características originais de df1:")
print(df1.columns)

# Perform one-hot encoding na coluna "feature 5"
df1 = pd.get_dummies(df1, columns=['feature 5'])

# Print as novas características do dataframe df1 após one-hot encoding
print("\nNovas características de df1 após one-hot encoding:")
print(df1.columns)

# Print as primeiras cinco linhas do dataframe df1
print("\nPrimeiras cinco linhas de df1 após one-hot encoding:")
print(df1.head())

