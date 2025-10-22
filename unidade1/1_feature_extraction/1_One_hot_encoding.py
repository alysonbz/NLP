import pandas as pd
from src.utils import load_df1_one_hot

df1 = load_df1_one_hot()

# Print the features of df1
print("Atributos originais:")
print(df1.columns.tolist())
print()

# Perform one-hot encoding on column "feature 5"
df1 = pd.get_dummies(df1, columns=["feature 5"])

# Print the new features of df1
print("Novos atributos após One-Hot Encoding:")
print(df1.columns.tolist())
print()

# Print first five rows of df1
print("Primeiras 5 linhas do novo DataFrame:")
print(df1.head())