import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from preprocessing import (
    normalizar_texto, remover_links, remover_mencoes, remover_numeros,
    tokenizar, lemmatizacao, remover_stopwords, stemming
)

from atribute_extraction import ExtratorAtributos


# ----------------------------------------------------------------------
# 🔹 PRÉ-PROCESSAMENTO
# ----------------------------------------------------------------------
def aplicar_preprocessamento(textos, usar_lemma=True):
    textos_processados = []

    for txt in textos:

        txt = normalizar_texto(txt)
        txt = remover_links(txt)
        txt = remover_mencoes(txt)
        txt = remover_numeros(txt)

        tokens = tokenizar(txt)

        tokens = lemmatizacao(tokens) if usar_lemma else stemming(tokens)

        tokens = remover_stopwords(tokens)

        textos_processados.append(tokens)

    return textos_processados


# ----------------------------------------------------------------------
# 🔹 TREINAR E AVALIAR (Naive Bayes)
# ----------------------------------------------------------------------
def treinar_e_avaliar(X_train, X_test, y_train, y_test):

    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    return accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test)


# ----------------------------------------------------------------------
# 🔹 EXPERIMENTOS A & B
# ----------------------------------------------------------------------
def executar_experimentos(df, coluna_texto, coluna_label):

    resultados_a = []

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[coluna_texto], df[coluna_label], test_size=0.2, random_state=42
    )

    for preprocess in [False, True]:

        # ----------------------------------------------------
        # PRÉ-PROCESSAMENTO
        # ----------------------------------------------------
        if preprocess:
            X_train_proc = aplicar_preprocessamento(X_train_raw, usar_lemma=True)
            X_test_proc = aplicar_preprocessamento(X_test_raw, usar_lemma=True)
        else:
            # Sem preprocessamento: tokenizar apenas por split
            X_train_proc = [txt.split() for txt in X_train_raw]
            X_test_proc = [txt.split() for txt in X_test_raw]

        # Extratores (separados para TREINO e TESTE)
        extrator_train = ExtratorAtributos(X_train_proc)
        extrator_test = ExtratorAtributos(X_test_proc)

        tecnicas = [
            ("Analise Estatística", "estat"),
            ("CountVectorizer", "bow"),
            ("TF-IDF", "tfidf"),
            ("Word2Vec", "w2v"),
        ]

        for nome, tipo in tecnicas:

            # ----------------------------------------------------
            # A) ANÁLISE ESTATÍSTICA
            # ----------------------------------------------------
            if tipo == "estat":
                Xtr = extrator_train.analise_estatistica().values
                Xte = extrator_test.analise_estatistica().values

            # ----------------------------------------------------
            # B) COUNT VECTORIZER
            # ----------------------------------------------------
            elif tipo == "bow":
                Xtr = extrator_train.cv_fit_transform()
                extrator_test.cv_vectorizer = extrator_train.cv_vectorizer
                Xte = extrator_test.cv_transform()

            # ----------------------------------------------------
            # C) TF-IDF
            # ----------------------------------------------------
            elif tipo == "tfidf":
                Xtr = extrator_train.tfidf_fit_transform()
                extrator_test.tfidf_vectorizer = extrator_train.tfidf_vectorizer
                Xte = extrator_test.tfidf_transform()

            # ----------------------------------------------------
            # D) WORD2VEC (treinado apenas no TREINO)
            # ----------------------------------------------------
            elif tipo == "w2v":
                extrator_train.word2vec_fit()
                extrator_test.w2v_model = extrator_train.w2v_model  # Copia o modelo para o teste

                Xtr = extrator_train.word2vec_transform()
                Xte = extrator_test.word2vec_transform()

                # Normalização
                scaler = MinMaxScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)

            # Treina e avalia
            _, acuracia = treinar_e_avaliar(Xtr, Xte, y_train, y_test)

            resultados_a.append({
                "Tecnica": nome,
                "Preprocessamento": "Com" if preprocess else "Sem",
                "Acuracia": acuracia,
            })

    df_resultados_a = pd.DataFrame(resultados_a)
    melhores = df_resultados_a[df_resultados_a["Preprocessamento"] == "Com"]

    return df_resultados_a, melhores


# ----------------------------------------------------------------------
# 🔹 ITEM C — Comparar LEMMA vs STEM por técnica
# ----------------------------------------------------------------------
def comparar_lemma_vs_stem(df, coluna_texto, coluna_label, tecnica):

    X_train, X_test, y_train, y_test = train_test_split(
        df[coluna_texto], df[coluna_label], test_size=0.2, random_state=42
    )

    resultados = []

    for modo in ["lemma", "stem"]:

        usar_lemma = (modo == "lemma")

        X_train_proc = aplicar_preprocessamento(X_train, usar_lemma)
        X_test_proc = aplicar_preprocessamento(X_test, usar_lemma)

        extrator_train = ExtratorAtributos(X_train_proc)
        extrator_test = ExtratorAtributos(X_test_proc)

        # ----------------------------------------------------
        # SELECIONAR TÉCNICA
        # ----------------------------------------------------
        if tecnica == "tfidf":
            Xtr = extrator_train.tfidf_fit_transform()
            extrator_test.tfidf_vectorizer = extrator_train.tfidf_vectorizer
            Xte = extrator_test.tfidf_transform()

        elif tecnica == "bow":
            Xtr = extrator_train.cv_fit_transform()
            extrator_test.cv_vectorizer = extrator_train.cv_vectorizer
            Xte = extrator_test.cv_transform()

        elif tecnica == "estat":
            Xtr = extrator_train.analise_estatistica().values
            Xte = extrator_test.analise_estatistica().values

        elif tecnica == "w2v":
            extrator_train.word2vec_fit()
            extrator_test.w2v_model = extrator_train.w2v_model

            Xtr = extrator_train.word2vec_transform()
            Xte = extrator_test.word2vec_transform()

            scaler = MinMaxScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)

        # ----------------------------------------------------
        # Avaliar
        # ----------------------------------------------------
        _, acuracia = treinar_e_avaliar(Xtr, Xte, y_train, y_test)

        resultados.append({
            "Tecnica": tecnica,
            "Metodo": modo,
            "Acuracia": acuracia
        })

    return pd.DataFrame(resultados)
