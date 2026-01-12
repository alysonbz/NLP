import pandas as pd
from preprocessing import preprocess_sentence

df = pd.read_parquet("../train-00000-of-00001.parquet")

# Escolhe uma linha qualquer
i = 10

premise_original = df.loc[i, "premise"]
hypothesis_original = df.loc[i, "hypothesis"]

premise_clean = preprocess_sentence(premise_original)
hypothesis_clean = preprocess_sentence(hypothesis_original)

print("===== EXEMPLO DE PRÉ-PROCESSAMENTO (PAR DE FRASES) =====")
print(f"Premise original ({i}):")
print(premise_original)

print("\nHypothesis original:")
print(hypothesis_original)

print("\nPremise após pré-processamento:")
print(premise_clean)

print("\nHypothesis após pré-processamento:")
print(hypothesis_clean)
print("=========================================================")
