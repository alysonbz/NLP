from enelvo.normaliser import Normaliser
import spacy
import nltk 

nltk.download('stopwords')

nlp = spacy.load("pt_core_news_sm")
nltk.download('rslp')

def corrigir(msg):
    norm = Normaliser(tokenizer='readable', capitalize_inis=True,
                      capitalize_pns=True, capitalize_acs=True,
                      sanitize=True)
    return norm.normalise(msg)

def remover_stopwords(msg):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    return [word for word in tokenizar(msg) if word not in stopwords]

def tokenizar(msg):
    doc = nlp(msg)
    return [token.text for token in doc]

def lemmatizar(msg):
    doc = nlp(msg)
    return [token.lemma_ for token in doc]

def stemmar(msg):
    stemmer = nltk.stem.RSLPStemmer()
    return [stemmer.stem(token) for token in tokenizar(msg)]

