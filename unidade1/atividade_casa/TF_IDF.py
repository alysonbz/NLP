import math

textos = [
    "this is not normal",
    "i am a dog",
    "i live in brazil",
    "brazil is a good country",
    "he lives in brazil",
]

def TF_IDF(corpus):
    N = len(corpus)
    vocab = set(x for x in " ".join(corpus).split(" "))

    # print(vocab)
    TF = [[0 for i in range(len(vocab))] for j in range(N)]
    # print(TF)
    for i in range(N): # linhas
        for j in range(len(vocab)): # cols
            doc = corpus[i]
            word = list(vocab)[j]
            count = 0
            for w in doc.split(" "):
                if w == word:
                    count +=1
            TF[i][j] = count/N

    # print(TF)
 
    def count_n_docs_with_term(term):
        total = 0
        for x in corpus:
            if term in x.split(" "):
                total += 1
        return total
    
    idfs = {x: math.log(N/count_n_docs_with_term(x)) for x in vocab}

TF_IDF(textos)