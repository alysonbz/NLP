# Regex úteis
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NON_LETTER_RE = re.compile(r"[^a-zA-ZÀ-ÿ\s]")


def to_lower(text: str) -> str:
    """Transforma tudo em minúsculo"""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove URLs do texto"""
    return URL_RE.sub(" ", text)


def remove_mentions(text: str) -> str:
    """Remove @menções (como em redes sociais)"""
    return MENTION_RE.sub(" ", text)


def remove_hashtags(text: str) -> str:
    """Remove #hashtags"""
    return HASHTAG_RE.sub(" ", text)


def remove_non_letters(text: str) -> str:
    """Remove números, emojis e símbolos, mantendo apenas letras e espaços"""
    return NON_LETTER_RE.sub(" ", text)


def basic_clean(text: str) -> str:
    """
    Limpeza básica:
    - minúsculas
    - remove URL, menções, hashtags
    - remove caracteres não alfabéticos
    - retira espaços extras
    """
    if not isinstance(text, str):
        text = str(text)

    text = to_lower(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_non_letters(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenização simples por espaço (após limpeza básica)"""
    text = basic_clean(text)
    if not text:
        return []
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove stopwords em português"""
    return [t for t in tokens if t not in stopwords_pt]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lematiza tokens usando spaCy (pt_core_news_sm)"""
    if not tokens:
        return []
    doc = nlp(" ".join(tokens))
    lemmas = [tok.lemma_ for tok in doc if tok.lemma_.strip()]
    return lemmas


def stem_tokens(tokens: List[str]) -> List[str]:
    """Aplica stemming usando RSLPStemmer (Português)"""
    if not tokens:
        return []
    return [stemmer.stem(t) for t in tokens]


def preprocess_text(text: str, mode: str = "lemma", remove_stops: bool = True) -> str:
    """
    Pré-processamento completo.
    mode:
      - "lemma": usa lematização
      - "stem": usa stemming
      - None / "none": apenas limpeza básica
    """
    tokens = tokenize(text)

    if remove_stops:
        tokens = remove_stopwords(tokens)

    if mode == "lemma":
        tokens = lemmatize_tokens(tokens)
    elif mode == "stem":
        tokens = stem_tokens(tokens)
    # se mode == "none", não modifica os tokens além da limpeza / stopwords

    return " ".join(tokens)


def preprocess_lemma(text: str) -> str:
    """Atalho: pré-processamento padrão com lematização"""
    return preprocess_text(text, mode="lemma", remove_stops=True)


def preprocess_stem(text: str) -> str:
    """Atalho: pré-processamento com stemming"""
    return preprocess_text(text, mode="stem", remove_stops=True)


def preprocess_no_clean(text: str) -> str:
    """
    Função de 'sem pré-processamento' para comparação:
    apenas garante que o texto é string.
    """
    if not isinstance(text, str):
        return str(text)
    return text
