
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
print(f'\nTexto corrigido: {resposta}\n')


doc = nlp(resposta)     # pegando já corrigida

# Token             TOKEN COM PALAVRAS
tk = []
for token in doc:
    tk.append(token.text)


# Stemming          RADICAL DA PALAVRA
stm = [stemmer.stem(token) for token in tk]
stem = []
for token in stm:
    stem.append(token)


# Lemmatização      VERBO NO INFINITIVO
lem = []
for token in doc:
    lem.append(token.lemma_)


# Stop_words        PALAVRAS QUE AFETAM O MODELO POR NÃO SER IMPORTANTE E SEREM MUITAS
stops = nlp.Defaults.stop_words
print(f'Os stop words padrões são: {list(stops)[:5]}\n')
veri_stops = [token.text.lower() in stops or token.is_punct for token in doc]


# Tabela
df = pd.DataFrame({'token': tk, 'stemming': stem, 'lemma': lem, 'stop_words': veri_stops})
print('\n',df)