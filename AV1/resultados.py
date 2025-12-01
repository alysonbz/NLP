import pandas as pd

from preprocessing import carregar_dataset
from atribute_extraction import ExtratorAtributos
from classification import executar_experimentos, comparar_lemma_vs_stem


print("\n--- Dataset ---")
df = carregar_dataset("AV1/b2w.csv")

# Remove registros sem rótulo
df = df.dropna(subset=["polarity"])

coluna_texto = "review_text"
coluna_label = "polarity"


# ITEM A — Experimentos COM x SEM pré-processamento
resultados_a, melhores_b = executar_experimentos(
    df,
    coluna_texto,
    coluna_label
)

print("\n--- Resultados item A ---")
print("Comparação COM x SEM pré-processamento:\n")
print(resultados_a)


# ITEM B — Melhores resultados com pré-processamento

print("\n--- Resultados item B ---")
print("Melhores resultados somente com pré-processamento:\n")
print(melhores_b)

# Identifica melhor técnica

melhor_tecnica = (
    melhores_b.sort_values(by="Acuracia", ascending=False)
    .iloc[0]["Tecnica"]
)

print("\nMelhor técnica encontrada no item B:", melhor_tecnica)

tecnica_cod = {
    "TF-IDF": "tfidf",
    "CountVectorizer": "bow"
}.get(melhor_tecnica, "estat")


# ITEM C — Comparação Lemma x Stemming
resultados_c = comparar_lemma_vs_stem(
    df,
    coluna_texto,
    coluna_label,
    tecnica=tecnica_cod
)

print("\n--- Resultados item C ---")
print("Comparação entre LEMMA x STEMMING usando a melhor técnica:\n")
print(resultados_c)

print("\nFIM DOS EXPERIMENTOS\n")