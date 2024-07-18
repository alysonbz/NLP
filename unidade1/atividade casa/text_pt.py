from nltk.stem import PorterStemmer, WordNetLemmatizer
from enelvo.normaliser import Normaliser
from nltk.corpus import stopwords
import pandas as pd
import nltk

nltk.download('punkt')
nltk.download('stopwords')

norm = Normaliser(tokenizer='readable')


# Faça uma implementação para que sua aplicação receba um texto com erros gramaticais em português:
# a) Realize a correção do texto
texto = 'Eu fui ao parke com minha familiya e brinqei muito!'
texto_corrigido = norm.normalise(texto)

print('texto original: ', texto)
print('texto corrigido: ', texto_corrigido)

# b) Tokenize o texto
tokens = nltk.word_tokenize(texto_corrigido)

# c) Remova as stop words
stop_words = set(stopwords.words('portuguese'))
tokens_sem_stopwords = [token for token in tokens if token.lower() not in stop_words]

# d) Faça uma tabela mostrando 3 colunas para todos os tokens: token , stemming e lematização
# stemming
stemmer = PorterStemmer()
tokens_stemming = [stemmer.stem(token) for token in tokens_sem_stopwords]

# lematização
lemmatizer = WordNetLemmatizer()
tokens_lem = [lemmatizer.lemmatize(token) for token in tokens_sem_stopwords]

df = pd.DataFrame({
    'tokens': [tokens],
    'stemming': [tokens_stemming],
    'lematização': [tokens_lem]
})

print(df.T)