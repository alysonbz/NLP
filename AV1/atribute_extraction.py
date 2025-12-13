class AttributeExtractor:
    """
    Classe que organiza diferentes formas de representação / exclusão de atributos:
    - CountVectorizer
    - TF-IDF
    - Matriz de coocorrência
    - Word2Vec (média dos vetores)
    - Seleção estatística (chi^2)
    """

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 1)):
        self.max_features = max_features
        self.ngram_range = ngram_range

        self.count_vect: Optional[CountVectorizer] = None
        self.tfidf_vect: Optional[TfidfVectorizer] = None
        self.cooc_vocab_: Optional[List[str]] = None
        self.cooc_matrix_: Optional[np.ndarray] = None
        self.w2v_model: Optional[Word2Vec] = None
        self.selector_: Optional[SelectKBest] = None

    # ============ COUNT VECTORIZER ============

    def fit_count(self, texts: List[str]) -> csr_matrix:
        self.count_vect = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )
        X = self.count_vect.fit_transform(texts)
        return X

    def transform_count(self, texts: List[str]) -> csr_matrix:
        if self.count_vect is None:
            raise RuntimeError("CountVectorizer não foi ajustado. Chame fit_count primeiro.")
        return self.count_vect.transform(texts)

    # ============ TF-IDF ============

    def fit_tfidf(self, texts: List[str]) -> csr_matrix:
        self.tfidf_vect = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )
        X = self.tfidf_vect.fit_transform(texts)
        return X

    def transform_tfidf(self, texts: List[str]) -> csr_matrix:
        if self.tfidf_vect is None:
            raise RuntimeError("TfidfVectorizer não foi ajustado. Chame fit_tfidf primeiro.")
        return self.tfidf_vect.transform(texts)

    # ============ MATRIZ DE COOCORRÊNCIA ============

    def fit_cooccurrence(self, texts: List[str]) -> np.ndarray:
        """
        Usa CountVectorizer para construir a matriz documento-termo X,
        depois calcula uma matriz de coocorrência aproximada: C = X^T X.
        """
        temp_vect = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )
        X = temp_vect.fit_transform(texts)
        vocab = temp_vect.get_feature_names_out()

        cooc = (X.T @ X).toarray().astype(np.float32)
        np.fill_diagonal(cooc, 0.0)

        self.cooc_vocab_ = list(vocab)
        self.cooc_matrix_ = cooc
        return cooc

    def get_cooccurrence(self) -> Tuple[List[str], np.ndarray]:
        if self.cooc_vocab_ is None or self.cooc_matrix_ is None:
            raise RuntimeError("Coocorrência ainda não foi ajustada. Chame fit_cooccurrence primeiro.")
        return self.cooc_vocab_, self.cooc_matrix_

    # ============ WORD2VEC ============

    def fit_word2vec(self, tokenized_texts: List[List[str]],
                     vector_size: int = 100,
                     window: int = 5,
                     min_count: int = 2,
                     workers: int = 4) -> None:
        """
        Treina um modelo Word2Vec nos textos já tokenizados.
        """
        self.w2v_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )

    def _sentence_to_vec(self, tokens: List[str]) -> np.ndarray:
        """
        Calcula o embedding médio dos tokens de uma sentença.
        """
        if self.w2v_model is None:
            raise RuntimeError("Word2Vec ainda não foi treinado. Chame fit_word2vec primeiro.")

        vectors = []
        for t in tokens:
            if t in self.w2v_model.wv:
                vectors.append(self.w2v_model.wv[t])

        if not vectors:
            return np.zeros(self.w2v_model.vector_size, dtype=np.float32)

        return np.mean(vectors, axis=0)

    def transform_word2vec(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """
        Gera um vetor para cada sentença (média dos embeddings das palavras).
        """
        if self.w2v_model is None:
            raise RuntimeError("Word2Vec ainda não foi treinado. Chame fit_word2vec primeiro.")

        emb_list = [self._sentence_to_vec(tokens) for tokens in tokenized_texts]
        return np.vstack(emb_list)

    # ============ ANÁLISE ESTATÍSTICA INDIVIDUAL (CHI^2) ============

    def fit_statistical_selector(self, X_train, y_train, k: int = 2000):
        """
        Seleção de atributos via chi^2 (análise estatística individual).
        X_train: matriz esparsa (ex: Count ou TF-IDF)
        y_train: rótulos
        """
        self.selector_ = SelectKBest(chi2, k=k)
        self.selector_.fit(X_train, y_train)

    def transform_with_selector(self, X):
        if self.selector_ is None:
            raise RuntimeError("Selector não ajustado. Chame fit_statistical_selector primeiro.")
        return self.selector_.transform(X)
