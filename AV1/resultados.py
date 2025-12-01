import pandas as pd

from preprocessing import carregar_dataset
from atribute_extraction import ExtratorAtributos
from classification import executar_experimentos, comparar_lemma_vs_stem


print("\n Dataset")
df = carregar_dataset("AV1/b2w.csv")

df = df.dropna(subset=["polarity"])

coluna_texto = "review_text"
coluna_label = "polarity"


resultados_a, melhores_b = executar_experimentos(df, coluna_texto, coluna_label)

print("\n Resultados item a)")
print("Comparação COM x SEM pré-processamento\n")
print(resultados_a)

print("\n Resultados item b)")
print("Melhores resultados usando APENAS pré-processamento\n")
print(melhores_b)


melhor_tecnica = (
    melhores_b.sort_values(by="Acuracia", ascending=False).iloc[0]["Tecnica"]
)

print("\n Melhor técnica encontrada no item B:", melhor_tecnica)

tecnica_cod = (
    "tfidf" if melhor_tecnica == "TF-IDF" else
    "bow" if melhor_tecnica == "CountVectorizer" else
    "estat"
)

resultados_c = comparar_lemma_vs_stem(
    df,
    coluna_texto,
    coluna_label,
    tecnica=tecnica_cod
)

print("\n Resultados item c)")
print("Comparação LEMMA x STEMMING usando a melhor técnica\n")
print(resultados_c)

print("\nFIM DOS EXPERIMENTOS\n")
