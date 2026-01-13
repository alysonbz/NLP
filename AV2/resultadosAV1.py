import pandas as pd
from AV1.preprocessing import carregar_dataset
from AV2.av1classf import executar_experimentos

print("\n--- Dataset ---")
df = carregar_dataset("AV1/b2w.csv")

# Remove registros sem rótulo
df = df.dropna(subset=["polarity"])

# Amostragem
df = df.sample(5000, random_state=42)

coluna_texto = "review_text"
coluna_label = "polarity"

#======================================
# EXECUTA OS EXPERIMENTOS
#=======================================
resultados, _ = executar_experimentos(df, coluna_texto, coluna_label)

#=======================================
# FILTRA SOMENTE COM PRÉ-PROCESSAMENTO
#=======================================
resultados_com_pre = resultados[resultados["Preprocessamento"] == "Com"]

print("\n--- RESULTADOS (COM PRÉ-PROCESSAMENTO) ---\n")
print(resultados_com_pre.to_string(index=False))

#=======================================
# MELHOR RESULTADO PELA ACURÁCIA
#=======================================
melhor = resultados_com_pre.sort_values(
    by="Acuracia", ascending=False
).iloc[0]

print("\n--- MELHOR RESULTADO (MAIOR ACURÁCIA) ---")
print(f"Técnica  : {melhor['Técnica']}")
print(f"Acurácia: {melhor['Acuracia']:.4f}")
print(f"Precisão: {melhor['Precisao']:.4f}")
print(f"Recall  : {melhor['Recall']:.4f}")
print(f"F1-score: {melhor['F1']:.4f}")
