import string

class matrizes_Corrcorrencia():
    def __init__(self, texto):
        self._texto = texto

    def definicao_corpus(self):
        textos_unicos = set()
        for i in self._texto:
            i = i.lower()
            i = i.translate(str.maketrans('','', string.punctuation))
            tokens = i.split()
            textos_unicos.update(tokens)

        self.corpus = sorted(textos_unicos)
        return self.corpus




