import pandas as pd
import re
import string
from enelvo.normaliser import Normaliser
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Certifique-se de que os recursos do NLTK estão baixados
nltk.download('punkt')
nltk.download('stopwords')

# Caminhos para os datasets
caminho_treino = r'C:/Users/bianc/Downloads/Ciencia_de_dados/PLN/NLP/AV1/train.csv'
caminho_teste = r'C:/Users/bianc/Downloads/Ciencia_de_dados/PLN/NLP/AV1/test.csv'

# Carregar datasets
redacao_treino = pd.read_csv(caminho_treino)
redacao_teste = pd.read_csv(caminho_teste)


def remove_bracketed_text(text):

    # Regex para encontrar padrões dentro de colchetes
    pattern = r'\[.*?\]'

    # Substitui os padrões encontrados por uma string vazia
    cleaned_text = re.sub(pattern, '', text)

    # Retorna o texto sem os padrões dentro dos colchetes
    return cleaned_text.strip()

# Função para remover pontuação
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
#$$$$$$
def to_lowercase(text):
    return text.lower()

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in all_stop_words]
    return ' '.join(filtered_words)


def remove_numbers(text):
    words = text.split()
    filtered_words = [word for word in words if not word.isdigit()]
    return ' '.join(filtered_words)





def preprocess_text(text):
    text = remove_bracketed_text(text)
    text = to_lowercase(text)
    #text = remove_numbers(text)
    #text = remove_stopwords(text)
    #text = replace_coloquialismos(text)
    text = remove_punctuation(text)


    return text

redacao_treino["aplicar"] = redacao_treino["essay"].apply(preprocess_text)

#redacao_treino.to_csv(r'C:/Users/bianc/Downloads/Ciencia_de_dados/PLN/NLP/AV1/train_preprocessed.csv', index=False)
print(redacao_treino[['essay','aplicar']])