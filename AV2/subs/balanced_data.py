from sklearn.utils import resample
import pandas as pd

# Carregar dataset
test_df = pd.read_csv("test_csv.csv")

# Separar classes
df_majority = test_df[test_df["is_hate_speech"] == 0]  # Classe majoritária (0)
df_minority = test_df[test_df["is_hate_speech"] == 1]  # Classe minoritária (1)

# Aplicar downsampling na classe 0
df_majority_downsampled = resample(
    df_majority,
    replace=False,  # Não permite duplicação
    n_samples=len(df_minority),  # Igualar à classe minoritária (2462)
    random_state=42  # Para reprodutibilidade
)

# Combinar classe minoritária com a classe 0 reduzida
train_df_balanced = pd.concat([df_majority_downsampled, df_minority])
train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
train_df_balanced.to_csv("test_csv_balanced.csv", index=False)
print(train_df_balanced["is_hate_speech"].value_counts())
