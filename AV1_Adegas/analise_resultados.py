# analise_resultados.py
# ---------------------
# Usa o modelo salvo em modelo_final.joblib
# para gerar e SALVAR a figura da matriz de confusão.

import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from classification import carregar_dataset_fakenews  # reutiliza a função


if __name__ == "__main__":
    # 1) Carregar dataset (mesma função e mesmos arquivos do classification.py)
    textos, labels = carregar_dataset_fakenews("News_fake.csv", "News_notFake.csv")

    # 2) Fazer o mesmo split treino/teste (mesmo random_state e stratify!)
    textos_treino, textos_teste, y_treino, y_teste = train_test_split(
        textos,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # 3) Carregar o modelo final salvo
    artefatos = joblib.load("modelo_final.joblib")
    tipo_atributo = artefatos["tipo_atributo"]
    preprocessamento = artefatos["preprocessamento"]
    extrator = artefatos["extrator"]
    classificador = artefatos["classificador"]

    print(f"Tipo de atributo do modelo final: {tipo_atributo}")
    print(f"Pré-processamento do modelo final: {preprocessamento}")

    # 4) Gerar X_teste usando apenas os MÉTODOS DE TRANSFORMAÇÃO
    #    (o extrator já está treinado/fitado no classification.py)
    if tipo_atributo == "count":
        X_teste = extrator.transformar_bow(textos_teste)
    elif tipo_atributo == "tfidf":
        X_teste = extrator.transformar_tfidf(textos_teste)
    elif tipo_atributo == "cooc":
        X_teste = extrator.transformar_coocorrencia(textos_teste)
    elif tipo_atributo == "word2vec":
        X_teste = extrator.transformar_word2vec(textos_teste)
    elif tipo_atributo == "stats":
        X_teste = extrator.atributos_estatisticos_simples(textos_teste)
    else:
        raise ValueError(f"Tipo de atributo desconhecido: {tipo_atributo}")

    # 5) Predições do modelo final
    y_pred = classificador.predict(X_teste)

    # 6) Matriz de confusão
    cm = confusion_matrix(y_teste, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

    disp.plot()
    plt.title("Matriz de confusão - Modelo final")
    plt.tight_layout()
    plt.savefig("fig_matriz_confusao_modelo_final.png", dpi=300)  # SALVA FIGURA
    plt.show()

    print("\nMatriz de confusão:")
    print(cm)
