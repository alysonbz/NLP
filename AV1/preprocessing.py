import re
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pattern.vector import stemmer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words_pt = set(stopwords.words('portuguese'))
stop_words_en = set(stopwords.words('english'))
all_stop_words = stop_words_pt.union(stop_words_en)
lemmatizer = WordNetLemmatizer()

# Dicionário de coloquialismos, gírias e suas substituições
coloquialismos = {"vc": "você","vcs": "vocês","tb": "também","pq": "porque","blz": "beleza","blza": "beleza","ñ": "não","n": "não",
                  "q": "que","tá": "está","cê": "você","eh": "é","bora": "vamos","vamo": "vamos","blza": "beleza","soh": "só","kd": "cadê",
                  "aki": "aqui","eh": "é","dps": "depois","pf": "por favor","hj": "hoje","mt": "muito","mto": "muito","flw": "falou",
                  "vlw": "valeu","d": "de","tao": "estão","tão": "estão","mano": "cara","bro": "cara","krl": "caramba","blw": "beleza",
                  "tamo": "estamos","to": "estou","só": "somente","vdd": "verdade","fml": "família","ctz": "certeza","sqn": "só que não",
                  "vcê": "você","tbm": "também","tpw": "tipo","qnd": "quando","vlw": "valeu","fechô": "fechou","papo": "conversa"}


def load_portuguese_hate_speech():
    return pd.read_csv('./dataset/portuguese_hate_speech.csv')


# Função para remover menções e links
def remove_mentions_and_links(text):
    text = re.sub(r'@\w+', '', text)  # Remove menções
    text = re.sub(r'http\S+', '', text)  # Remove links
    return text


# Função para converter o texto para minúsculas
def to_lowercase(text):
    return text.lower()


# Função para remover letras isoladas
def remove_single_letters(text):
    words = text.split()
    filtered_words = [word for word in words if len(word) > 1]
    return ' '.join(filtered_words)


# Função para remover stopwords
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in all_stop_words]
    return ' '.join(filtered_words)


# Função para substituir ou remover coloquialismos
def replace_coloquialismos(text):
    words = text.split()
    replaced_words = [coloquialismos.get(word, word) for word in words]
    return ' '.join(replaced_words)


# Função para remover números
def remove_numbers(text):
    words = text.split()
    filtered_words = [word for word in words if not word.isdigit()]
    return ' '.join(filtered_words)

# Função para remover pontuação
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Função para aplicar stemming
def apply_stemming(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)


# Função para aplicar lematização
def apply_lemmatization(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


# Função para aplicar todos os pré-processamentos
def preprocess_text(text):
    text = remove_mentions_and_links(text)
    text = to_lowercase(text)
    text = remove_numbers(text)
    text = remove_single_letters(text)
    text = remove_stopwords(text)
    text = replace_coloquialismos(text)
    text = remove_punctuation(text)
    return text


