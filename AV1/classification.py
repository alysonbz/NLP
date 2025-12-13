def load_X_y_from_df(df: pd.DataFrame,
                     text_col: str = "review_text",
                     label_col: str = "polarity") -> Tuple[pd.Series, pd.Series]:

    df_clean = df[[text_col, label_col]].copy()

    # Remove valores ausentes no texto ou no rótulo
    df_clean = df_clean.dropna(subset=[text_col, label_col])

    # Converte rótulos para inteiro
    df_clean[label_col] = df_clean[label_col].astype(int)

    X_text = df_clean[text_col].astype(str)
    y = df_clean[label_col]

    return X_text, y


def train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test) -> float:
    """
    Treina LogisticRegression e retorna acurácia
    """
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    return acc


def experiment_a(X_text, y):
    """
    (a) Usando todas as formas de remoção de atributos,
    compare resultados com pré-processamento e sem pré-processamento
    """
    results = []

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    preproc_options = {
        "sem_preproc": preprocess_no_clean,
        "lemma": preprocess_lemma,
    }

    for preproc_name, preproc_fun in preproc_options.items():
        print(f"=== Pré-processamento: {preproc_name} ===")
        X_train_pp = X_train_raw.apply(preproc_fun)
        X_test_pp = X_test_raw.apply(preproc_fun)

        extractor = AttributeExtractor(max_features=5000, ngram_range=(1, 2))

        # 1) CountVectorizer
        X_train_count = extractor.fit_count(list(X_train_pp))
        X_test_count = extractor.transform_count(list(X_test_pp))
        acc_count = train_and_evaluate(X_train_count, X_test_count, y_train, y_test)
        results.append({
            "preproc": preproc_name,
            "features": "Count",
            "accuracy": acc_count
        })

        # 2) TF-IDF
        X_train_tfidf = extractor.fit_tfidf(list(X_train_pp))
        X_test_tfidf = extractor.transform_tfidf(list(X_test_pp))
        acc_tfidf = train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test)
        results.append({
            "preproc": preproc_name,
            "features": "TF-IDF",
            "accuracy": acc_tfidf
        })

        # 3) TF-IDF + seleção chi²
        extractor.fit_statistical_selector(X_train_tfidf, y_train, k=2000)
        X_train_tfidf_sel = extractor.transform_with_selector(X_train_tfidf)
        X_test_tfidf_sel = extractor.transform_with_selector(X_test_tfidf)
        acc_tfidf_sel = train_and_evaluate(X_train_tfidf_sel, X_test_tfidf_sel, y_train, y_test)
        results.append({
            "preproc": preproc_name,
            "features": "TF-IDF + chi^2",
            "accuracy": acc_tfidf_sel
        })

        # 4) Word2Vec
        tokenized_train = [tokenize(t) for t in X_train_pp]
        tokenized_test = [tokenize(t) for t in X_test_pp]
        extractor.fit_word2vec(tokenized_train, vector_size=100, window=5, min_count=2)
        X_train_w2v = extractor.transform_word2vec(tokenized_train)
        X_test_w2v = extractor.transform_word2vec(tokenized_test)
        acc_w2v = train_and_evaluate(X_train_w2v, X_test_w2v, y_train, y_test)
        results.append({
            "preproc": preproc_name,
            "features": "Word2Vec",
            "accuracy": acc_w2v
        })

    results_df = pd.DataFrame(results)
    print("\n=== Resultados Questão 3(a) ===")
    display(results_df.pivot(index="features", columns="preproc", values="accuracy"))
    return results_df


def experiment_b(X_text, y, best_preproc: str = "lemma"):
    """
    (b) Compare as formas de remoção de atributos usando o pré-processamento escolhido.
    """
    preproc_map = {
        "lemma": preprocess_lemma,
        "sem_preproc": preprocess_no_clean,
        "stem": preprocess_stem,
    }

    preproc_fun = preproc_map[best_preproc]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_pp = X_train_raw.apply(preproc_fun)
    X_test_pp = X_test_raw.apply(preproc_fun)

    extractor = AttributeExtractor(max_features=5000, ngram_range=(1, 2))
    results = []

    # Count
    X_train_count = extractor.fit_count(list(X_train_pp))
    X_test_count = extractor.transform_count(list(X_test_pp))
    acc_count = train_and_evaluate(X_train_count, X_test_count, y_train, y_test)
    results.append({"features": "Count", "accuracy": acc_count})

    # TF-IDF
    X_train_tfidf = extractor.fit_tfidf(list(X_train_pp))
    X_test_tfidf = extractor.transform_tfidf(list(X_test_pp))
    acc_tfidf = train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test)
    results.append({"features": "TF-IDF", "accuracy": acc_tfidf})

    # TF-IDF + chi²
    extractor.fit_statistical_selector(X_train_tfidf, y_train, k=2000)
    X_train_tfidf_sel = extractor.transform_with_selector(X_train_tfidf)
    X_test_tfidf_sel = extractor.transform_with_selector(X_test_tfidf)
    acc_tfidf_sel = train_and_evaluate(X_train_tfidf_sel, X_test_tfidf_sel, y_train, y_test)
    results.append({"features": "TF-IDF + chi^2", "accuracy": acc_tfidf_sel})

    # Word2Vec
    tokenized_train = [tokenize(t) for t in X_train_pp]
    tokenized_test = [tokenize(t) for t in X_test_pp]
    extractor.fit_word2vec(tokenized_train, vector_size=100, window=5, min_count=2)
    X_train_w2v = extractor.transform_word2vec(tokenized_train)
    X_test_w2v = extractor.transform_word2vec(tokenized_test)
    acc_w2v = train_and_evaluate(X_train_w2v, X_test_w2v, y_train, y_test)
    results.append({"features": "Word2Vec", "accuracy": acc_w2v})

    results_df = pd.DataFrame(results)
    print("\n=== Resultados Questão 3(b) (pré-processamento:", best_preproc, ") ===")
    display(results_df)
    return results_df


def experiment_c(X_text, y, best_features: str = "TF-IDF + chi^2"):
    """
    (c) Compare lematização vs stemming usando a melhor forma de atributos (best_features)
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    preproc_modes = {
        "lemma": preprocess_lemma,
        "stem": preprocess_stem,
    }

    results = []

    for mode_name, preproc_fun in preproc_modes.items():
        print(f"=== Comparando modo: {mode_name} com features: {best_features} ===")
        X_train_pp = X_train_raw.apply(preproc_fun)
        X_test_pp = X_test_raw.apply(preproc_fun)

        extractor = AttributeExtractor(max_features=5000, ngram_range=(1, 2))

        if best_features == "Count":
            X_train_v = extractor.fit_count(list(X_train_pp))
            X_test_v = extractor.transform_count(list(X_test_pp))

        elif best_features == "TF-IDF":
            X_train_v = extractor.fit_tfidf(list(X_train_pp))
            X_test_v = extractor.transform_tfidf(list(X_test_pp))

        elif best_features == "TF-IDF + chi^2":
            X_train_tfidf = extractor.fit_tfidf(list(X_train_pp))
            X_test_tfidf = extractor.transform_tfidf(list(X_test_pp))
            extractor.fit_statistical_selector(X_train_tfidf, y_train, k=2000)
            X_train_v = extractor.transform_with_selector(X_train_tfidf)
            X_test_v = extractor.transform_with_selector(X_test_tfidf)

        elif best_features == "Word2Vec":
            tokenized_train = [tokenize(t) for t in X_train_pp]
            tokenized_test = [tokenize(t) for t in X_test_pp]
            extractor.fit_word2vec(tokenized_train, vector_size=100, window=5, min_count=2)
            X_train_v = extractor.transform_word2vec(tokenized_train)
            X_test_v = extractor.transform_word2vec(tokenized_test)

        else:
            raise ValueError("best_features inválido")

        acc = train_and_evaluate(X_train_v, X_test_v, y_train, y_test)
        results.append({
            "preproc_mode": mode_name,
            "features": best_features,
            "accuracy": acc
        })

    results_df = pd.DataFrame(results)
    print("\n=== Resultados Questão 3(c) ===")
    display(results_df)
    return results_df

# Carregar X e y a partir do DataFrame
X_text, y = load_X_y_from_df(df)

# Questão 3(a)
res_a = experiment_a(X_text, y)

# >>> Aqui se usa o pré-processamento que foi escolhido no item a)
best_preproc = "lemma"

# Questão 3(b)
res_b = experiment_b(X_text, y, best_preproc=best_preproc)

# >>> Aqui se escolhe a melhor forma de atributos com base em res_b
best_features = "TF-IDF"

# Questão 3(c)
res_c = experiment_c(X_text, y, best_features=best_features)

# Salvar resultados
res_a.to_csv("resultados_q3a.csv", index=False)
res_b.to_csv("resultados_q3b.csv", index=False)
res_c.to_csv("resultados_q3c.csv", index=False)
