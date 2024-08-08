# Faça uma implementação para que sua aplicação receba um texto com erros gramaticais em português:
# a) Realize a correção do texto
# b) Tokenize o texto
# c) Remova as stop words
# d) Faça uma tabela mostrando 3 colunas para todos os tokens: token , stemming e lematização

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

frase = "Quem não chora não mama!"

tokens = word_tokenize(frase.lower())

nlp = spacy.load("pt_core_news_sm")
doc = nlp(frase)

lemmas = [token.lemma_ for token in doc]

stemmed_lemmas = [stemmer.stem(token) for token in tokens]

stop_words = set(stopwords.words('portuguese'))

tokens_no_stop = []
lemmas_no_stop = []
stemmed_lemmas_no_stop = []

for token, lemma, stemmed_lemma in zip(tokens, lemmas, stemmed_lemmas):
    if token not in stop_words:
        tokens_no_stop.append(token)
        lemmas_no_stop.append(lemma)
        stemmed_lemmas_no_stop.append(stemmed_lemma)


print("{:<15} {:<15} {:<15}".format("Token", "Lematização", "Stemming"))
print("-" * 45)
for token, lemma, stemmed_lemma in zip(tokens_no_stop, lemmas_no_stop, stemmed_lemmas_no_stop):
    print("{:<15} {:<15} {:<15}".format(token, lemma, stemmed_lemma))
