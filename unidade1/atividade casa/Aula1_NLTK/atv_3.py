
# 3. Faça uma implementação para que sua aplicação receba um texto com erros gramaticais em português.
#    a) Realize a correção do texto
#    b) Tokenize o texto
#    c) Remova as stop words
#    d) Faça uma tabela mostrando 3 colunas para todos os tokens: token, stemming e lemmatização

import nltk
from nltk.stem import RSLPStemmer
import pandas as pd
import spacy
from enelvo.normaliser import Normaliser
nlp = spacy.load("pt_core_news_sm")
#nltk.download('rslp')  # rodar uma vez
stemmer = RSLPStemmer()   # somente para deixar o stemming em português
# Corretor
normalizador = Normaliser(tokenizer='readable', capitalize_pns = True, capitalize_acs = True, capitalize_inis = True, sanitize = True)
msg = 'a maria foi ao shopp pq estava trsite, acho q hj foi uma dia dificiu.'
resposta = normalizador.normalise(msg)
print('\nTexto corrigido:', resposta)
doc = nlp(resposta)     # pegando já corrigida
# Token
tk = []
for token in doc:
    tk.append(token.text)
# Stemming
stm = [stemmer.stem(token) for token in tk]
stem = []
for token in stm:
    stem.append(token)
# Lemmatização
lem = []
for token in doc:
    lem.append(token.lemma_)
# Tabela
df = pd.DataFrame({'token': tk, 'stemming': stem, 'lemma': lem})
print('\n',df)











