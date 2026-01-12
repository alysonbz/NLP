import pandas as pd

# carregar previsões
df = pd.read_csv("av2/outputs/q1_predictions.csv")

# criar coluna de acerto
df["correct"] = df["y_true"] == df["y_pred_llm"]

# separar acertos e erros
acertos = df[df["correct"] == True]
erros = df[df["correct"] == False]

print("\n===== EXEMPLOS DE ACERTOS =====\n")
for _, row in acertos.sample(min(2, len(acertos)), random_state=42).iterrows():
    print("Sentence pair ID:", row["sentence_pair_id"])
    print("Label real:", row["y_true"])
    print("Predição LLM:", row["y_pred_llm"])
    print("-" * 60)

print("\n===== EXEMPLOS DE ERROS =====\n")
for _, row in erros.sample(min(3, len(erros)), random_state=42).iterrows():
    print("Sentence pair ID:", row["sentence_pair_id"])
    print("Label real:", row["y_true"])
    print("Predição LLM:", row["y_pred_llm"])
    print("-" * 60)
