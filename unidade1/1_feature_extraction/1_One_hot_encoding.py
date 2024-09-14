import pandas as pd
from src.utils import load_df1_one_hot

# Load the dataframe df1
df1 = load_df1_one_hot()

# Print the features of df1

print("Original features of df1:", df1.columns)

# Perform one-hot encoding on column "feature 5"
df1_encoded = pd.get_dummies(df1, columns=["feature 5"])

# Print the new features of df1

print("New features of df1 after one-hot encoding:", df1_encoded.columns)

# Print first five rows of df1

print("First 5 rows of df1:")
print(df1_encoded.head())
