def manual_tf_idf(corpus):
    """
    Esta função recebe uma lista de textos (corpus) e retorna uma matriz TF-IDF.

    Parâmetros:
    corpus (list): Lista de strings, onde cada string é um documento/texto.

    Retorna:
    vocab (dict): Um dicionário que mapeia cada palavra única para seu índice na matriz.
    tf_idf_matrix (numpy.ndarray): Uma matriz onde cada linha representa um documento e
                                   cada coluna representa o valor TF-IDF de uma palavra específica.
    """
    # Criação do vocabulário a partir do corpus
    vocab = {}
    for text in corpus:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    # Inicializar as matrizes de TF e DF
    tf_matrix = np.zeros((len(corpus), len(vocab)))
    df_vector = np.zeros(len(vocab))

    # Calcular TF (Term Frequency) e DF (Document Frequency)
    for i, text in enumerate(corpus):
        word_count = len(text.split())
        word_freq = {}

        for word in text.split():
            if word in vocab:
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1

        for word, count in word_freq.items():
            index = vocab[word]
            tf_matrix[i, index] = count / word_count
            df_vector[index] += 1

    #  Calcular IDF (Inverse Document Frequency)
    idf_vector = np.log(len(corpus) / (df_vector + 1)) + 1

    # Calcular a matriz TF-IDF
    tf_idf_matrix = tf_matrix * idf_vector

    return tf_idf_matrix