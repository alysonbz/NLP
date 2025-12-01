"""" 
Neste exercicio você deve utilizar um único classifiador para aplicar no seu dataset, de acordo com a label escolhida.
Voce deve comparar os resultados quantitiativos nos seguintes casos:

a) Utilizando todas as formas de extração de atributos, compare os resultados de acerto do classificador com pré-processamento e sem pré-processamento. Mostre as taxas para os casos de forma organizada.

b) Compare as formas de extração de atributos, usando com pré-processamento que você escolheu no item a)

C) Faça um variação de dois pré-procesamentos que compare lemmatização e steming, considerando a melhor forma de extração de atributos vista no item b)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Função de preprocessamento
from preprocessing import preprocessar  
from atribute_extraction import ExtratorAtributos


def treinar_e_avaliar(X_train, X_test, y_train, y_test):

    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    return accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test)


def gerar_atributos(tipo, extrator_train, extrator_test, pretrained_w2v=None):
    
    # ANALISE ESTATÍSTICA
    if tipo == "estat":
        Xtr = extrator_train.analise_estatistica().values
        Xte = extrator_test.analise_estatistica().values
        return Xtr, Xte

    # BAG OF WORDS

    if tipo == "bow":
        Xtr = extrator_train.cv_fit_transform()
        extrator_test.cv_vectorizer = extrator_train.cv_vectorizer
        Xte = extrator_test.cv_transform()
        return Xtr, Xte

    # TF-IDF

    if tipo == "tfidf":
        Xtr = extrator_train.tfidf_fit_transform()
        extrator_test.tfidf_vectorizer = extrator_train.tfidf_vectorizer
        Xte = extrator_test.tfidf_transform()
        return Xtr, Xte

    # WORD2VEC

    if tipo == "w2v":
        
        if pretrained_w2v is None:
            extrator_train.word2vec_fit()
        else:
            extrator_train.w2v_model = pretrained_w2v

        extrator_test.w2v_model = extrator_train.w2v_model

        Xtr = extrator_train.word2vec_transform()
        Xte = extrator_test.word2vec_transform()

        scaler = MinMaxScaler()
        return scaler.fit_transform(Xtr), scaler.transform(Xte)

    raise ValueError("Técnica desconhecida.")


#a e b

def executar_experimentos(df, coluna_texto, coluna_label, pretrained_w2v=None):

    resultados = []

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[coluna_texto], df[coluna_label], test_size=0.2, random_state=42
    )

    tecnicas = [
        ("Analise Estatística", "estat"),
        ("CountVectorizer", "bow"),
        ("TF-IDF", "tfidf"),
        ("Word2Vec", "w2v"),
    ]

    for usar_pre in [False, True]:

        # 1) PRÉ-PROCESSAMENTO
        if usar_pre:
            X_train_proc = [preprocessar(txt) for txt in X_train_raw]
            X_test_proc = [preprocessar(txt) for txt in X_test_raw]
        else:
            X_train_proc = [txt.lower().split() for txt in X_train_raw]
            X_test_proc = [txt.lower().split() for txt in X_test_raw]

        extrator_train = ExtratorAtributos(X_train_proc)
        extrator_test = ExtratorAtributos(X_test_proc)

        # 2) EXECUTAR TODAS AS TÉCNICAS
        for nome, tipo in tecnicas:

            Xtr, Xte = gerar_atributos(tipo, extrator_train, extrator_test, pretrained_w2v)

            _, acuracia = treinar_e_avaliar(Xtr, Xte, y_train, y_test)

            resultados.append({
                "Técnica": nome,
                "Preprocessamento": "Com" if usar_pre else "Sem",
                "Acuracia": acuracia,
            })

    df_resultados = pd.DataFrame(resultados)

    melhores = df_resultados[df_resultados["Preprocessamento"] == "Com"]

    return df_resultados, melhores


#c
def comparar_lemma_vs_stem(df, coluna_texto, coluna_label, tecnica, pretrained_w2v=None):

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[coluna_texto], df[coluna_label], test_size=0.2, random_state=42
    )

    resultados = []

    for metodo in ["lemma", "stem"]:

        X_train_proc = [preprocessar(txt, aplicar_lemma=(metodo == "lemma")) for txt in X_train_raw]
        X_test_proc = [preprocessar(txt, aplicar_lemma=(metodo == "lemma")) for txt in X_test_raw]

        extrator_train = ExtratorAtributos(X_train_proc)
        extrator_test = ExtratorAtributos(X_test_proc)

        # Gera atributos da técnica desejada
        Xtr, Xte = gerar_atributos(tecnica, extrator_train, extrator_test, pretrained_w2v)

        _, acuracia = treinar_e_avaliar(Xtr, Xte, y_train, y_test)

        resultados.append({
            "Técnica": tecnica,
            "Método": metodo,
            "Acuracia": acuracia
        })

    return pd.DataFrame(resultados)
