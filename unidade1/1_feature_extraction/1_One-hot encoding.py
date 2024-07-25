import pandas as pd

df1 = pd.read_csv(r'C:\Users\joaod\Documents\faculdade-ciencia-de-dados\Semestres\2024_1\NLP\unidade1\dataset\df1_unidade1.csv')

# Print the features of df1
print(df1.columns)

# Perform one-hot encoding
df1 = pd.get_dummies(df1, columns=['feature 5'])

# Print the new features of df1
print(df1.columns)

# Print first five rows of df1
print(df1.head())