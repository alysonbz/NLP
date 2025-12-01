
# 1) Lê dois arquivos CSV:
#       - News_fake.csv    -> notícias falsas
#       - News_notFake.csv -> notícias verdadeiras
# 2) Usa a coluna "title" como texto
# 3) Cria uma coluna "label": 1 = fake, 0 = not fake
# 4) Divide em treino e teste
# 5) Usa uma Regressão Logística como classificador
# 6) Roda:
#    (a) comparação COM x SEM pré-processamento para cada tipo de atributo
#    (b) comparação entre os tipos de atributo com pré-processamento fixo
#    (c) comparação lematização x stemming com o melhor tipo de atributo
# 7) Treina um modelo final com a melhor combinação e SALVA o modelo + extrator



import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # para salvar o modelo e o extrator

from atribute_extraction import ExtratorAtributos



# Carrega os dois CSVs e monta o dataset
def carregar_dataset_fakenews(caminho_fake: str, caminho_real: str):
    """
    Retorna:
        - lista de textos (titles)
        - vetor de labels (0 ou 1)
    """
    # Lê as notícias falsas
    df_fake = pd.read_csv(caminho_fake)
    df_fake["label"] = 1  # 1 = fake

    # Lê as notícias verdadeiras
    df_real = pd.read_csv(caminho_real)
    df_real["label"] = 0  # 0 = not fake

    # Junta tudo em um único DataFrame
    df = pd.concat([df_fake, df_real], ignore_index=True)

    # Usa o título como texto
    textos = df["title"].astype(str).tolist()
    labels = df["label"].values

    return textos, labels



# Função simples para treinar e avaliar o classificador
def treinar_avaliar(X_treino, X_teste, y_treino, y_teste) -> float:
    """
    Treina uma Regressão Logística nos dados de treino
    e retorna a acurácia no conjunto de teste.
    """
    classificador = LogisticRegression(max_iter=1000)
    classificador.fit(X_treino, y_treino)
    y_pred = classificador.predict(X_teste)
    acuracia = accuracy_score(y_teste, y_pred)
    return acuracia



# Geração de atributos para cada tipo
def gerar_atributos(
    extrator: ExtratorAtributos,
    tipo_atributo: str,
    textos_treino,
    textos_teste,
):
    """
    - "count"    -> Bag-of-Words
    - "tfidf"    -> TF-IDF
    - "cooc"     -> coocorrência (bigramas)
    - "word2vec" -> Word2Vec
    - "stats"    -> estatísticas simples
    """
    if tipo_atributo == "count":
        X_treino = extrator.ajustar_bow(textos_treino)
        X_teste = extrator.transformar_bow(textos_teste)

    elif tipo_atributo == "tfidf":
        X_treino = extrator.ajustar_tfidf(textos_treino)
        X_teste = extrator.transformar_tfidf(textos_teste)

    elif tipo_atributo == "cooc":
        X_treino = extrator.ajustar_coocorrencia(textos_treino)
        X_teste = extrator.transformar_coocorrencia(textos_teste)

    elif tipo_atributo == "word2vec":
        X_treino = extrator.ajustar_word2vec(textos_treino)
        X_teste = extrator.transformar_word2vec(textos_teste)

    elif tipo_atributo == "stats":
        X_treino = extrator.atributos_estatisticos_simples(textos_treino)
        X_teste = extrator.atributos_estatisticos_simples(textos_teste)

    else:
        raise ValueError(f"Tipo de atributo desconhecido: {tipo_atributo}")

    return X_treino, X_teste



# main
if __name__ == "__main__":
    caminho_fake = "News_fake.csv"
    caminho_real = "News_notFake.csv"

    # Carrega textos e labels
    textos, labels = carregar_dataset_fakenews(caminho_fake, caminho_real)

    textos_treino, textos_teste, y_treino, y_teste = train_test_split(
        textos,
        labels,
        test_size=0.2,      # 20% para teste
        random_state=42,    # semente fixa (reproduzível)
        stratify=labels,    # mantém proporção das classes
    )

    tipos_atributos = ["count", "tfidf", "cooc", "word2vec", "stats"]


    # a) COM x SEM pré-processamento para cada tipo
    print("\na) COM x SEM pré-processamento")
    resultados_a = []

    for tipo in tipos_atributos:
        # SEM PRÉ-PROCESSAMENTO
        extrator_sem = ExtratorAtributos(
            usar_preprocessamento=False
        )
        X_treino_sem, X_teste_sem = gerar_atributos(
            extrator_sem, tipo, textos_treino, textos_teste
        )
        acc_sem = treinar_avaliar(X_treino_sem, X_teste_sem, y_treino, y_teste)
        resultados_a.append(
            {
                "tipo_atributo": tipo,
                "preprocessamento": "sem",
                "acuracia": acc_sem,
            }
        )

        # COM PRÉ-PROCESSAMENTO (com lematização)
        extrator_com = ExtratorAtributos(
            usar_preprocessamento=True,
            remover_sw=True,
            usar_stem=False,
            usar_lemma=True,
        )
        X_treino_com, X_teste_com = gerar_atributos(
            extrator_com, tipo, textos_treino, textos_teste
        )
        acc_com = treinar_avaliar(X_treino_com, X_teste_com, y_treino, y_teste)
        resultados_a.append(
            {
                "tipo_atributo": tipo,
                "preprocessamento": "com",
                "acuracia": acc_com,
            }
        )

    df_a = pd.DataFrame(resultados_a)
    print(df_a)

    df_a.to_csv("resultados_experimento_a.csv", index=False)


    # PLOT
    tabela_a = df_a.pivot(
        index="tipo_atributo",
        columns="preprocessamento",
        values="acuracia",
    )
    ax_a = tabela_a.plot(kind="bar")
    plt.title("Experimento a): Acurácia COM x SEM pré-processamento")
    plt.xlabel("Tipo de atributo")
    plt.ylabel("Acurácia")
    plt.ylim(0, 1)  # acurácia vai de 0 a 1
    plt.legend(title="Pré-processamento")
    plt.tight_layout()
    plt.savefig("fig_exp_a_preprocessamento.png", dpi=300)
    plt.show()


    # b) Comparar formas de extração de atributos
    print("\nb) Comparação entre tipos de atributos (com um pré-processamento fixo)")
    resultados_b = []

    extrator_b = ExtratorAtributos(
        usar_preprocessamento=True,
        remover_sw=True,
        usar_stem=False,
        usar_lemma=True,
    )

    for tipo in tipos_atributos:
        X_treino_b, X_teste_b = gerar_atributos(
            extrator_b, tipo, textos_treino, textos_teste
        )
        acc_b = treinar_avaliar(X_treino_b, X_teste_b, y_treino, y_teste)
        resultados_b.append(
            {
                "tipo_atributo": tipo,
                "preprocessamento": "lemma",
                "acuracia": acc_b,
            }
        )

    df_b = pd.DataFrame(resultados_b)
    print(df_b)

    df_b.to_csv("resultados_experimento_b.csv", index=False)


    # PLOT
    plt.figure()
    plt.bar(df_b["tipo_atributo"], df_b["acuracia"])
    plt.title("Experimento b): Acurácia por tipo de atributo (com lematização)")
    plt.xlabel("Tipo de atributo")
    plt.ylabel("Acurácia")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("fig_exp_b_tipos_atributos.png", dpi=300)  # SALVA FIGURA
    plt.show()


    # escolher automaticamente o melhor tipo de atributo com base no experimento b)
    idx_melhor = df_b["acuracia"].idxmax()
    tipo_melhor = df_b.loc[idx_melhor, "tipo_atributo"]
    print(f"\nMelhor tipo de atributo no experimento (b): {tipo_melhor}")


    # c) Lematização x Stemming
    print(f"\nc) Lematização x Stemming usando '{tipo_melhor}'")
    resultados_c = []

    # ----- LEMATIZAÇÃO -----
    extrator_lemma = ExtratorAtributos(
        usar_preprocessamento=True,
        remover_sw=True,
        usar_stem=False,
        usar_lemma=True,
    )
    X_treino_lemma, X_teste_lemma = gerar_atributos(
        extrator_lemma, tipo_melhor, textos_treino, textos_teste
    )
    acc_lemma = treinar_avaliar(X_treino_lemma, X_teste_lemma, y_treino, y_teste)
    resultados_c.append(
        {
            "tipo_atributo": tipo_melhor,
            "preprocessamento": "lematizacao",
            "acuracia": acc_lemma,
        }
    )

    # STEMMING
    extrator_stem = ExtratorAtributos(
        usar_preprocessamento=True,
        remover_sw=True,
        usar_stem=True,
        usar_lemma=False,
    )
    X_treino_stem, X_teste_stem = gerar_atributos(
        extrator_stem, tipo_melhor, textos_treino, textos_teste
    )
    acc_stem = treinar_avaliar(X_treino_stem, X_teste_stem, y_treino, y_teste)
    resultados_c.append(
        {
            "tipo_atributo": tipo_melhor,
            "preprocessamento": "stemming",
            "acuracia": acc_stem,
        }
    )

    df_c = pd.DataFrame(resultados_c)
    print(df_c)


    df_c.to_csv("resultados_experimento_c.csv", index=False)


    # PLOT
    plt.figure()
    plt.bar(df_c["preprocessamento"], df_c["acuracia"])
    plt.title(f"Experimento (c): Lematização x Stemming ({tipo_melhor})")
    plt.xlabel("Tipo de pré-processamento")
    plt.ylabel("Acurácia")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("fig_exp_c_lemma_vs_stem.png", dpi=300)  # SALVA FIGURA
    plt.show()


    # 5) Treinar modelo final e salvar para usar depois

    # Escolhe o melhor pré-processamento com base no experimento c)
    idx_melhor_prep = df_c["acuracia"].idxmax()
    melhor_prep = df_c.loc[idx_melhor_prep, "preprocessamento"]
    print(f"\nMelhor pré-processamento no experimento (c): {melhor_prep}")

    # Define flags de stemming/lematização para o modelo final
    if melhor_prep == "lematizacao":
        usar_stem_final = False
        usar_lemma_final = True
    elif melhor_prep == "stemming":
        usar_stem_final = True
        usar_lemma_final = False
    else:
        usar_stem_final = False
        usar_lemma_final = True

    # Cria um extrator final com a melhor configuração
    extrator_final = ExtratorAtributos(
        usar_preprocessamento=True,
        remover_sw=True,
        usar_stem=usar_stem_final,
        usar_lemma=usar_lemma_final,
    )

    # Gera atributos com o tipo e pré-processamento escolhidos
    X_treino_final, X_teste_final = gerar_atributos(
        extrator_final, tipo_melhor, textos_treino, textos_teste
    )

    # Treina o classificador final
    classificador_final = LogisticRegression(max_iter=1000)
    classificador_final.fit(X_treino_final, y_treino)
    y_pred_final = classificador_final.predict(X_teste_final)
    acuracia_final = accuracy_score(y_teste, y_pred_final)

    print(
        f"\nAcurácia do modelo final ({tipo_melhor} + {melhor_prep}): "
        f"{acuracia_final:.4f}"
    )

    # Salva o extrator + classificador em um único arquivo para usar depois
    artefatos = {
        "tipo_atributo": tipo_melhor,
        "preprocessamento": melhor_prep,
        "extrator": extrator_final,
        "classificador": classificador_final,
    }

    joblib.dump(artefatos, "modelo_final.joblib")
    print("\nModelo final salvo em 'modelo_final.joblib'")
