import spacy
from spacy.lang.pt.examples import sentences
from enelvo.normaliser import Normaliser
from nltk.corpus import stopwords
import nltk
import pandas as pd
from nltk.stem import RSLPStemmer


text =  "No meu café da manhã, tinha sobre a meza, quejo, prezunto, mortandela, matega, saucinha e iogurte natural. Mas o café estava sem asúcar e eu presizo de uma colher para mecher o café. Era tanta coisa que não sobrava espaso na meza. Liguei a televisam e estava paçando o “Bom Dia São Paulo”, onde mostrou como se constrói o espaso geográfico. Os home construimdo nos morros, as caza de simento e madera."
# Faça uma implementação para que sua aplicação receba um texto com erros gramaticais em português:
# a) Realize a correção do texto
norm = Normaliser(tokenizer='readable', capitalize_inis=True,capitalize_pns=True , capitalize_acs=True,sanitize= True)
resp = norm.normalise(text)
print(resp)
# b) Tokenize o texto
nlp = spacy.load('pt_core_news_sm')
doc = nlp(resp)

for token in doc:
    print(token.text)

# c) Remova as stop words
stop_words = nltk.corpus.stopwords.words('portuguese')
tokens_sem_stopwords = [palavra for palavra in resp.split() if palavra.lower() not in stop_words]
texto_final = ' '.join(tokens_sem_stopwords)

print("Texto final sem stopwords:", texto_final)
# d) Faça uma tabela mostrando 3 colunas para todos os tokens: token , stemming e lematização

# Stemming usando o RSLPStemmer do NLTK (Português)
stemmer = RSLPStemmer()
tokens_stemming = [stemmer.stem(token) for token in tokens_sem_stopwords]

# Lematização usando spaCy
tokens_lemmas = [nlp(token)[0].lemma_ for token in tokens_sem_stopwords]

# Criar o DataFrame com os resultados
df = pd.DataFrame({
    'Token': tokens_sem_stopwords,
    'Stemming': tokens_stemming,
    'Lematização': tokens_lemmas
})

print(df)