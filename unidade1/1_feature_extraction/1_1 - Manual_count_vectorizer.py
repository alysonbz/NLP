from src.utils import load_movie_review_clean_dataset

df = load_movie_review_clean_dataset()


def count_vectorizer(list_to_vect):
    # 1 - Tokenizar: Transformando cada sentença em listas de palavras
    tokenized = []
    for sent in list_to_vect:
        words = sent.lower().split()
        tokenized.append(words)

    # 2 - Criando vocabulário (saco de palavras)
    vocab = {}
    i = 0
    for words in tokenized:
        for w in words:
            if w not in vocab:
                vocab[w]= i
                i += 1

    # 3 - Criando matriz de contagem
    # matriz com 0: n_sentencas x n_vocab
    matrix = [[0]*len(vocab) for _ in range(len(tokenized))]

    # preenchendo as contagens
    for sent_idx, words in enumerate(tokenized):
        for w in words:
            col = vocab[w]
            matrix[sent_idx][col] += 1
    return matrix, vocab

#teste
print(df['review'].head())
mat, vocab = count_vectorizer(df['review'])
print(mat)
print(vocab)

# teste
test_list = ["bom filme", "filme ruim", "filme bom bom"]
mat, vocab = count_vectorizer(test_list)
print("VOCAB:", vocab)
print("MATRIZ:")
for row in mat:
    print(row)




# tokenizar - pode ser com biblioteca
# criar saco de palavra (id para cada termo)
# criar a matriz de dicionario para cada sentença
# seu código aqui
