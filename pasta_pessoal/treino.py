import polars as pl
import nltk
from nltk.stem import WordNetLemmatizer # Lematização
from nltk.tokenize import word_tokenize # Tokenização
from nltk.tokenize import RegexpTokenizer # Tokenização 2
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("wordnet")

# Textos (docs)
tx1 = """
Se estiver se sentindo desmotivado ou sentindo que não é bom o suficiente
Incendeie o seu coração, enchugue as lagrimas e siga em frente
Quando se entristecer ou se acovardar lembre-se que o fluxo do tempo nunca para
Ele não vai te esperar enquanto você se afoga em tristeza.
    """

tx2 = """
Eu não quero que vc fique angustiado com a minha partida
Não esqueça que sou um hashira e que vou proteger vocês de onde eu estiver
Os novos botões precisam desabrochar, qualquer outro hashira pensaria da mesma forma
Porque eu acredito no talento de cada um de vocês.
         """

## PARTE 1 : TOKENIZAÇÃO

# Com word_tokenize [tx1]
tx2_tokens = word_tokenize(tx2)

# Com RegexpTokenizer
tokeniser = RegexpTokenizer(r'\w+') # alphanumeric tokens
tx1_tokens = tokeniser.tokenize(tx1)


# Normalização [tx1]
## Lematização

lemmatiser = WordNetLemmatizer()
lemmas = [lemmatiser.lemmatize(token.lower(), pos = 'v') for token in tx1_tokens]

# Steam
sno = nltk.stem.SnowballStemmer('portuguese')
steams = [sno.stem(token) for token in lemmas]

# Stopwords
stopwords_pt = stopwords.words("portuguese")

words = [word for word in steams if not word in stopwords_pt]
print(words)


# fonte : https://towardsdatascience.com/introduction-to-nlp-part-1-preprocessing-text-in-python-8f007d44ca96