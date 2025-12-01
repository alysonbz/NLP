

# - Atributos estatísticos simples
# - Bag-of-Words (CountVectorizer)
# - TF-IDF
# - Coocorrência (bigramas)
# - Word2Vec (média dos vetores por texto)


from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

from preprocessing import (
    preprocessar_corpus,
    tem_mencao,
    contar_stopwords_texto,
)


class ExtratorAtributos:
    """
    - usar_preprocessamento: se True, aplica pré-processamento nos textos
    - remover_sw: se True, remove stopwords
    - usar_stem: se True, aplica stemming
    - usar_lemma: se True, aplica lematização
    """



    def __init__(
        self,
        usar_preprocessamento: bool = True,
        remover_sw: bool = True,
        usar_stem: bool = False,
        usar_lemma: bool = False,
    ):
        self.usar_preprocessamento = usar_preprocessamento
        self.remover_sw = remover_sw
        self.usar_stem = usar_stem
        self.usar_lemma = usar_lemma

        # Estes objetos serão "treinados" (fit) nos dados de treino
        self.vetorizador_bow = None
        self.vetorizador_tfidf = None
        self.vetorizador_cooc = None
        self.modelo_word2vec = None



    # Função interna para preparar textos (pré-processar)
    def _preparar_textos(self, textos: List[str]) -> List[str]:
        """
        Se usar_preprocessamento=True, aplica o preprocessar_corpus.
        Caso contrário, só garante que tudo é string.
        """
        textos = [str(t) for t in textos]

        if not self.usar_preprocessamento:
            return textos

        # Aplica o pré-processamento definido no construtor
        return preprocessar_corpus(
            textos,
            remover_sw=self.remover_sw,
            usar_stem=self.usar_stem,
            usar_lemma=self.usar_lemma,
        )



    # Atributos estatísticos simples por texto
    def atributos_estatisticos_simples(self, textos: List[str]) -> np.ndarray:
        """
        - tamanho em caracteres
        - nº de palavras
        - nº de palavras únicas
        - tem menção @? (0 ou 1)
        - nº de stopwords
        Retorna uma matriz (n_textos, 5).
        """
        textos = [str(t) for t in textos]
        n = len(textos)

        tam_chars = np.zeros(n)
        num_palavras = np.zeros(n)
        num_unicas = np.zeros(n)
        flag_mencao = np.zeros(n)
        num_sw = np.zeros(n)

        for i, txt in enumerate(textos):
            # tamanho em caracteres
            tam_chars[i] = len(txt)

            # palavras separadas por espaço (simples)
            palavras = txt.split()
            num_palavras[i] = len(palavras)
            num_unicas[i] = len(set(palavras))

            # 1 se tiver @, 0 caso contrário
            flag_mencao[i] = 1 if tem_mencao(txt) else 0

            # nº de stopwords
            num_sw[i] = contar_stopwords_texto(txt)

        # Junta tudo em uma matriz de atributos
        X = np.column_stack(
            [tam_chars, num_palavras, num_unicas, flag_mencao, num_sw]
        )
        return X



    # Bag-of-Words (CountVectorizer)
    def ajustar_bow(self, textos: List[str]):
        """
        Treina (fit) o Bag-of-Words (CountVectorizer) usando os textos de TREINO
        e devolve a matriz de atributos X_treino.
        """
        textos_proc = self._preparar_textos(textos)
        self.vetorizador_bow = CountVectorizer()
        X = self.vetorizador_bow.fit_transform(textos_proc)
        return X



    def transformar_bow(self, textos: List[str]):
        """
        Aplica o Bag-of-Words já treinado nos textos de TESTE.
        """
        textos_proc = self._preparar_textos(textos)
        X = self.vetorizador_bow.transform(textos_proc)
        return X



    # TF-IDF
    def ajustar_tfidf(self, textos: List[str]):
        """
        Treina (fit) o TF-IDF usando os textos de TREINO.
        """
        textos_proc = self._preparar_textos(textos)
        self.vetorizador_tfidf = TfidfVectorizer()
        X = self.vetorizador_tfidf.fit_transform(textos_proc)
        return X



    def transformar_tfidf(self, textos: List[str]):
        """
        Aplica o TF-IDF já treinado nos textos de TESTE.
        """
        textos_proc = self._preparar_textos(textos)
        X = self.vetorizador_tfidf.transform(textos_proc)
        return X



    # Coocorrência (bigramas)
    def ajustar_coocorrencia(self, textos: List[str]):
        """
        Cria uma matriz de coocorrência simples usando BIGRAMAS,
        que são pares de palavras que aparecem juntos.
        """
        textos_proc = self._preparar_textos(textos)
        self.vetorizador_cooc = CountVectorizer(ngram_range=(2, 2))
        X = self.vetorizador_cooc.fit_transform(textos_proc)
        return X



    def transformar_coocorrencia(self, textos: List[str]):
        """
        Aplica a matriz de coocorrência (bigramas) já treinada nos textos de teste.
        """
        textos_proc = self._preparar_textos(textos)
        X = self.vetorizador_cooc.transform(textos_proc)
        return X



    # Word2Vec (média dos vetores)
    def ajustar_word2vec(self, textos: List[str], tamanho_vetor: int = 100):
        """
        Treina um modelo Word2Vec usando os textos de treino
        e gera um vetor por texto tirando a média dos vetores das palavras.
        """
        textos_proc = self._preparar_textos(textos)

        # Transforma cada texto em lista de palavras (simples: split)
        listas_tokens = [t.split() for t in textos_proc]

        # Treina o modelo Word2Vec
        self.modelo_word2vec = Word2Vec(
            sentences=listas_tokens,
            vector_size=tamanho_vetor,
            window=5,
            min_count=1,
            workers=4,
        )

        # Constrói a matriz de vetores dos documentos
        X = self._documentos_para_vetores(listas_tokens)
        return X



    def transformar_word2vec(self, textos: List[str]):
        """
        Gera vetores Word2Vec para os textos de TESTE,
        usando o modelo Word2Vec já treinado.
        """
        textos_proc = self._preparar_textos(textos)
        listas_tokens = [t.split() for t in textos_proc]
        X = self._documentos_para_vetores(listas_tokens)
        return X



    def _documentos_para_vetores(self, listas_tokens: List[List[str]]) -> np.ndarray:
        """
        Função interna para transformar cada lista de tokens
        em um único vetor (média dos vetores das palavras).
        """
        dim = self.modelo_word2vec.vector_size
        X = np.zeros((len(listas_tokens), dim))

        for i, tokens in enumerate(listas_tokens):
            vetores = []
            for palavra in tokens:
                if palavra in self.modelo_word2vec.wv:
                    vetores.append(self.modelo_word2vec.wv[palavra])

            if len(vetores) > 0:
                X[i] = np.mean(vetores, axis=0)
            # Se não tiver nenhuma palavra conhecida, o vetor fica como zeros

        return X
