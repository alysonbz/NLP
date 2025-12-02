# download completo do nltk: python -m nltk.downloader all


import re
from typing import List

import nltk
from nltk.corpus import stopwords

# Faz o download dos recursos do NLTK
nltk.download("punkt")
nltk.download("stopwords")

# Carrega a lista de stopwords em português
STOPWORDS_PT = set(stopwords.words("portuguese"))

# Tenta carregar o spaCy para lematização
try:
    import spacy
    NLP_PT = spacy.load("pt_core_news_sm")
except Exception:
    NLP_PT = None



def limpar_texto(texto: str) -> str:
    """
    Faz a limpeza básica do texto:
    - deixa tudo minúsculo
    - remove links (URLs)
    - remove menções (@usuario)
    - remove hashtags (#palavra)
    - remove números
    - remove pontuação
    - tira espaços repetidos
    """

    # deixa tudo minúsculo
    texto = texto.lower()

    # remove URLs (http..., https..., www...)
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)

    # remove menções do tipo @usuario
    texto = re.sub(r"@\w+", " ", texto)

    # remove hashtags do tipo #algo
    texto = re.sub(r"#\w+", " ", texto)

    # remove números
    texto = re.sub(r"\d+", " ", texto)

    # remove pontuação (tudo que não é letra, número ou espaço)
    texto = re.sub(r"[^\w\s]", " ", texto)

    # troca vários espaços por um só
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto



def tokenizar(texto: str) -> List[str]:
    """
    Separa o texto em palavras (tokens).
    Aqui usamos o tokenizer do NLTK configurado para português.
    """
    tokens = nltk.word_tokenize(texto, language="portuguese")
    return tokens



def remover_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove as stopwords (palavras muito comuns, como "de", "a", "o").
    """
    return [t for t in tokens if t not in STOPWORDS_PT]



def aplicar_stemming(tokens: List[str]) -> List[str]:
    """
    Aplica stemming (radicalização) nas palavras.
    Exemplo: "correndo", "correu" -> "corr".
    """
    from nltk.stem import SnowballStemmer

    stemmer = SnowballStemmer("portuguese")
    tokens_stem = [stemmer.stem(t) for t in tokens]
    return tokens_stem



def aplicar_lematizacao(tokens: List[str]) -> List[str]:
    """
    Aplica lematização (forma canônica da palavra).
    Exemplo: "correndo", "correu" -> "correr".

    Se o spaCy não estiver instalado, apenas devolve os tokens originais.
    """
    if NLP_PT is None:
        # Sem spaCy, não temos lematização de verdade.
        # Neste caso, apenas retornamos os tokens.
        return tokens

    # Junta tokens em uma frase para o spaCy
    doc = NLP_PT(" ".join(tokens))
    lemas = [token.lemma_ for token in doc]
    return lemas



def preprocessar_texto(
    texto: str,
    remover_sw: bool = True,
    usar_stem: bool = False,
    usar_lemma: bool = False,
) -> str:
    """
    Pré-processa UM texto com os seguintes passos:

    1) limpeza básica (limpar_texto)
    2) tokenização (tokenizar)
    3) remoção de stopwords (opcional)
    4) stemming OU lematização (opcionais)

    Retorna o texto final como string com as palavras separadas por espaço.
    """

    # 1) limpeza básica
    texto = limpar_texto(texto)

    # 2) tokenização
    tokens = tokenizar(texto)

    # 3) remove stopwords (se selecionado)
    if remover_sw:
        tokens = remover_stopwords(tokens)

    # 4) stemming ou lematização (não use os dois ao mesmo tempo)
    if usar_stem:
        tokens = aplicar_stemming(tokens)
    elif usar_lemma:
        tokens = aplicar_lematizacao(tokens)

    # Junta de volta em uma string
    texto_processado = " ".join(tokens)
    return texto_processado



def preprocessar_corpus(
    textos: List[str],
    remover_sw: bool = True,
    usar_stem: bool = False,
    usar_lemma: bool = False,
) -> List[str]:
    """
    Aplica o preprocessar_texto para uma lista de textos.
    Ou seja, pré-processa o "corpus" inteiro.
    """
    return [
        preprocessar_texto(
            t, remover_sw=remover_sw, usar_stem=usar_stem, usar_lemma=usar_lemma
        )
        for t in textos
    ]



def tem_mencao(texto: str) -> bool:
    """
    Verifica se o texto tem alguma menção do tipo @usuario.
    Essa função pode ser usada como um atributo extra.
    """
    return bool(re.search(r"@\w+", texto))



def contar_stopwords_texto(texto: str) -> int:
    """
    Conta quantas stopwords existem no texto.
    Aqui fazemos uma limpeza simples de pontuação antes.
    """
    # remove pontuação e deixa minúsculo
    texto_limpo = re.sub(r"[^\w\s]", " ", texto.lower())
    tokens = tokenizar(texto_limpo)
    contagem = sum(1 for t in tokens if t in STOPWORDS_PT)
    return contagem
