import pandas as pd
from src.utils import load_df1_one_hot
import pandas as pd
df1 = load_df1_one_hot()

# Print the features of df1
# Assuming df1 is already defined
print(df1.info())
print(df1.describe())
print(df1.head())


# Perform one-hot encoding on column "feature 5"
df1_encoded= pd.get_dummies(df1, columns=['feature 5'])

# Print the new features of df1
print(df1_encoded.info())


# Print first five rows of df1
print(df1_encoded.head())
