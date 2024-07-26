import pandas as pd
from src.utils import load_df1_one_hot

# Load the DataFrame
df1 = load_df1_one_hot()

# Print the features (columns) of df1
print("Features of df1:")
print(df1.columns)
