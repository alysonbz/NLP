from src.utils import load_movie_review_clean_dataset
import numpy as np



corpus = load_movie_review_clean_dataset()

def manual_count_vectorizer(corpus):
    """
    Esta função recebe uma lista de textos (corpus) e retorna uma matriz de contagem de palavras.

    Parâmetros:
    corpus (list): Lista de strings, onde cada string é um documento/texto.

    Retorna:
    vocab (dict): Um dicionário que mapeia cada palavra única para seu índice na matriz.
    count_matrix (numpy.ndarray): Uma matriz onde cada linha representa um documento e
                                  cada coluna representa a contagem de uma palavra específica.
    """
    # Criação do vocabulário a partir do corpus
    vocab = {}
    for text in corpus:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    # Inicializar a matriz de contagem
    count_matrix = np.zeros((len(corpus), len(vocab)))

    # Preencher a matriz de contagem com o número de ocorrências de cada palavra em cada documento
    for i, text in enumerate(corpus):
        for word in text.split():
            if word in vocab:
                count_matrix[i, vocab[word]] += 1

    return count_matrix