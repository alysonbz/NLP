import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocessing import (
    remover_uppercase, remover_stopwords,
    stemming, lemmatization, remover_simbolos,
    remover_pontuacao, remover_numeros, normalizar_espacos
)

from attribute_extraction import ExtratorManual, Extratores
from sklearn.preprocessing import StandardScaler


"""Neste exercicio você deve utilizar um único classifiador para aplicar no seu
dataset, de acordo com a label escolhida.
Voce deve comparar os resultados quantitiativos nos seguintes casos:

a) Utilizando todas as formas de extração de atributos, compare os resultados de
acerto do classificador com pré-processamento e sem pré-processamento.
Mostre as taxas para os casos de forma organizada.

b) Compare as formas de extração de atributos, usando com pré-processamento que
você escolheu no item a)

C) Faça um variação de dois pré-procesamentos que compare lemmatização e steming,
considerando a melhor forma de extração de atributos vista no item b)"""


def aplicar_preprocessamento(df, funcs):
    df = df.copy()
    for f in funcs:
        df["essay"] = df["essay"].astype(str).apply(f)
    return df

def avaliar_modelo(X_train, y_train, X_test, y_test, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=6000,
        solver="saga",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

def experimento_item_A(train_df, test_df, label_col):
    """
    Item (a):
    Comparar as formas de pré-processamento
    usando APENAS o Extratores (CountVectorizer interno).
    """

    resultados = []

    y_train = train_df[label_col]
    y_test = test_df[label_col]



    # -------- 1) SEM PREPROCESSAMENTO --------
    extr = Extratores()

    X_train = extr.cv_fit_transform(train_df)
    X_test = extr.cv_transform(test_df)

    acc = avaliar_modelo(X_train, y_train, X_test, y_test)
    resultados.append(("Sem preprocessamento", acc))


    # -------- 2) REMOVER UPPERCASE --------
    df_train = train_df.copy()
    df_test = test_df.copy()

    df_train["essay"] = df_train["essay"].apply(remover_uppercase)
    df_test["essay"]  = df_test["essay"].apply(remover_uppercase)

    extr = Extratores()
    X_train = extr.cv_fit_transform(df_train)
    X_test = extr.cv_transform(df_test)

    acc = avaliar_modelo(X_train, y_train, X_test, y_test)
    resultados.append(("Remover maiúsculas", acc))


    # -------- 3) REMOVER SÍMBOLOS --------
    df_train = train_df.copy()
    df_test = test_df.copy()

    df_train["essay"] = df_train["essay"].apply(remover_simbolos)
    df_test["essay"]  = df_test["essay"].apply(remover_simbolos)

    extr = Extratores()
    X_train = extr.cv_fit_transform(df_train)
    X_test = extr.cv_transform(df_test)

    acc = avaliar_modelo(X_train, y_train, X_test, y_test)
    resultados.append(("Remover símbolos", acc))


    # -------- 4) REMOVER STOPWORDS --------
    df_train = train_df.copy()
    df_test = test_df.copy()

    df_train["essay"] = df_train["essay"].apply(remover_stopwords)
    df_test["essay"]  = df_test["essay"].apply(remover_stopwords)

    extr = Extratores()
    X_train = extr.cv_fit_transform(df_train)
    X_test = extr.cv_transform(df_test)

    acc = avaliar_modelo(X_train, y_train, X_test, y_test)
    resultados.append(("Remover stopwords", acc))


    # -------- 5) REMOVER PONTUAÇÃO --------
    df_train = train_df.copy()
    df_test = test_df.copy()

    df_train["essay"] = df_train["essay"].apply(remover_pontuacao)
    df_test["essay"]  = df_test["essay"].apply(remover_pontuacao)

    extr = Extratores()
    X_train = extr.cv_fit_transform(df_train)
    X_test = extr.cv_transform(df_test)

    acc = avaliar_modelo(X_train, y_train, X_test, y_test)
    resultados.append(("Remover pontuação", acc))

    # STEMMING
    df_train = train_df.copy()
    df_test = test_df.copy()

    df_train["essay"] = df_train["essay"].apply(stemming)
    df_test["essay"]  = df_test["essay"].apply(stemming)

    extr = Extratores()
    X_train = extr.cv_fit_transform(df_train)
    X_test = extr.cv_transform(df_test)
    acc = avaliar_modelo(X_train, y_train, X_test, y_test)
    resultados.append(("Stemming", acc))

    # LEMMATIZATION
    df_train = train_df.copy()
    df_test = test_df.copy()

    df_train["essay"] = df_train["essay"].apply(lemmatization)
    df_test["essay"]  = df_test["essay"].apply(lemmatization)

    extr = Extratores()
    X_train = extr.cv_fit_transform(df_train)
    X_test = extr.cv_transform(df_test)

    acc = avaliar_modelo(X_train, y_train, X_test, y_test)
    resultados.append(("Lemmatização", acc))

    # NORMALIZAR ESPAÇOS
    df_train = train_df.copy()
    df_test = test_df.copy()

    df_train["essay"] = df_train["essay"].apply(normalizar_espacos)
    df_test["essay"]  = df_test["essay"].apply(normalizar_espacos)

    extr = Extratores()
    X_train = extr.cv_fit_transform(df_train)
    X_test = extr.cv_transform(df_test)

    acc = avaliar_modelo(X_train, y_train, X_test, y_test)
    resultados.append(("Normalizar espaços", acc))

    # -------- 6) PIPELINE COMPLETA --------
    preprocess_all = [
        remover_uppercase,
        remover_simbolos,
        remover_stopwords,
        remover_pontuacao,
        normalizar_espacos,
        stemming,
        lemmatization,
    ]

    df_train = train_df.copy()
    df_test = test_df.copy()

    for func in preprocess_all:
        df_train["essay"] = df_train["essay"].apply(func)
        df_test["essay"]  = df_test["essay"].apply(func)

    extr = Extratores()
    X_train = extr.cv_fit_transform(df_train)
    X_test = extr.cv_transform(df_test)

    acc = avaliar_modelo(X_train, y_train, X_test, y_test)
    resultados.append(("Pipeline completa", acc))

    return pd.DataFrame(resultados, columns=["Preprocessamento", "Acurácia"])


def experimento_item_B(train_df, test_df, label_col):
    """
    Item (b):
    Comparar TODAS as formas de extração de atributos,
    usando o pré-processamento escolhido no item (a).
    """

    resultados = []

    y_train = train_df[label_col]
    y_test = test_df[label_col]

    preprocess_funcs = [
        remover_stopwords,
    ]

    df_train_pre = train_df.copy()
    df_test_pre  = test_df.copy()

    for func in preprocess_funcs:
        df_train_pre["essay"] = df_train_pre["essay"].apply(func)
        df_test_pre["essay"]  = df_test_pre["essay"].apply(func)


    # --- 1) Extrator Manual ---
    extr_m = ExtratorManual()
    X_train_m = extr_m.criar_df_atributos(df_train_pre)
    X_test_m = extr_m.criar_df_atributos(df_test_pre)
    
    acc = avaliar_modelo(X_train_m, y_train, X_test_m, y_test, scale=True)
    resultados.append(("Manual COM preprocessing", acc))


    # --- 2) Bag-of-Words ---
    extr_cv = Extratores(text_col="essay")
    X_train_cv = extr_cv.cv_fit_transform(df_train_pre)
    X_test_cv = extr_cv.cv_transform(df_test_pre)

    acc = avaliar_modelo(X_train_cv, y_train, X_test_cv, y_test)
    resultados.append(("BoW COM preprocessing", acc))


    # --- 3) TF-IDF ---
    extr_tfidf = Extratores(text_col="essay")
    X_train_tfidf = extr_tfidf.tfidf_fit_transform(df_train_pre)
    X_test_tfidf = extr_tfidf.tfidf_transform(df_test_pre)

    acc = avaliar_modelo(X_train_tfidf, y_train, X_test_tfidf, y_test)
    resultados.append(("TF-IDF COM preprocessing", acc))


    # --- 4) Embeddings ---
    try:
        extr_emb = Extratores(text_col="essay")
        if extr_emb.embedding_model is not None:
            X_train_emb = extr_emb.embedding_transform(df_train_pre)
            X_test_emb = extr_emb.embedding_transform(df_test_pre)

            acc = avaliar_modelo(X_train_emb, y_train, X_test_emb, y_test)
            resultados.append(("Embeddings COM preprocessing", acc))
    except Exception as e:
        print("Falha ao carregar embeddings:", e)

    return pd.DataFrame(resultados, columns=["Método", "Acurácia"])

def experimento_item_C(train_df, test_df, label):
    """
    Item (c):
    Comparar lemmatização vs stemming usando a 
    MELHOR forma de extração do item B (Extrator Manual).
    """

    resultados = []

    y_train = train_df[label]
    y_test = test_df[label]

    # ======================
    #       STEMMING
    # ======================
    df_train_stem = train_df.copy()
    df_test_stem  = test_df.copy()

    df_train_stem["essay"] = df_train_stem["essay"].astype(str).apply(stemming)
    df_test_stem["essay"]  = df_test_stem["essay"].astype(str).apply(stemming)

    # novo extrator para não vazar vocabulário
    extr_stem = ExtratorManual()
    X_train_stem = extr_stem.criar_df_atributos(df_train_stem)
    X_test_stem  = extr_stem.criar_df_atributos(df_test_stem)

    acc_stem = avaliar_modelo(X_train_stem, y_train, X_test_stem, y_test, scale=True)
    resultados.append(("Stemming + ExtratorManual", acc_stem))


    # ======================
    #     LEMMATIZAÇÃO
    # ======================
    df_train_lem = train_df.copy()
    df_test_lem  = test_df.copy()

    df_train_lem["essay"] = df_train_lem["essay"].astype(str).apply(lemmatization)
    df_test_lem["essay"]  = df_test_lem["essay"].astype(str).apply(lemmatization)

    # novo extrator independente
    extr_lem = ExtratorManual()
    X_train_lem = extr_lem.criar_df_atributos(df_train_lem)
    X_test_lem  = extr_lem.criar_df_atributos(df_test_lem)

    acc_lem = avaliar_modelo(X_train_lem, y_train, X_test_lem, y_test, scale=True)
    resultados.append(("Lemmatização + ExtratorManual", acc_lem))


    # ======================
    #       RESULTADO
    # ======================
    return pd.DataFrame(resultados, columns=["Método", "Acurácia"])
