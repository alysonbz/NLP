from enelvo.normaliser import Normaliser
from nltk.stem import RSLPStemmer
import pandas as pd
import spacy
import nltk

nltk.download("stopwords")
nltk.download('rslp')

nlp = spacy.load("pt_core_news_sm")


texto = "hj eu fui para a igreja, foi muito legau"
print("Texto original:\n", texto)


normalizar = Normaliser(tokenizer="readable", capitalize_acs=True, 
                        capitalize_inis=True, capitalize_pns=True,
                        sanitize=True)

correcao = normalizar.normalise(texto)
print("\nTexto corrigido:\n", correcao, "\n")

doc = nlp(correcao)
for token in doc:
    print(token.text)

stopwords = nltk.corpus.stopwords.words("portuguese")

sem_stopwords = " ".join([palavra for palavra in correcao.split() if palavra.lower() not in stopwords])

print("\nTexto sem stopwords:\n", sem_stopwords, "\n")

stemmer = RSLPStemmer()

tabela = []
for token in doc:
    token_text = token.text
    token_stem = stemmer.stem(token_text)
    token_lemma = token.lemma_
    tabela.append([token_text, token_stem, token_lemma])

# DataFrame
df = pd.DataFrame(tabela, columns=["Token", "Stemming", "Lematização"])
print(df)


