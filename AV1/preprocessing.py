import re
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import wordnet

# Inicializando objetos necessários para processamento
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('portuguese'))
stemmatizer = RSLPStemmer()

# Adicionar stopwords manualmente
stop_words.update(['neste', 'nesta', 'nestes', 'nestas', 'r'])  # Adicione as que você precisar


def preprocess_with_lemmatization(text):
    # Tokeniza o texto e converte para minúsculas
    tokens = nltk.word_tokenize(text.lower(), language='portuguese')

    # Lematiza cada token que é alfanumérico e não está na lista de stopwords
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]

    # Junta os tokens lematizados de volta em uma string
    return ' '.join(lemmatized_tokens)

def preprocess_with_stemming(text):
    stemmer = RSLPStemmer()
    tokens = nltk.word_tokenize(text.lower(), language='portuguese')
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(stemmed_tokens)

def preprocess_text(text):
    """
    Função principal para pré-processar o texto.
    Aplica as funções de lematização, remoção de stopwords e menções.
    """
    # Converter para minúsculas
    text = text.lower()

    # Remover pontuação
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remover menções de números
    text = re.sub(r'\d+', '', text)

    # Tokenizar o texto
    tokens = nltk.word_tokenize(text, language='portuguese')

    # Remover stopwords e aplicar lematização
    processed_tokens = []
    for token in tokens:
        if token not in stop_words:
            stemmatize_token = stemmatizer.stem(token)
            processed_tokens.append(stemmatize_token)

    # Juntar tokens de volta em uma string
    processed_text = ' '.join(processed_tokens)

    return processed_text

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_sentence = words.copy()
    random_words = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_words)
    num_replaced = 0
    for random_word in random_words:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_sentence = [synonym if word == random_word else word for word in new_sentence]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_sentence)

def random_insertion(sentence, n=1):
    words = sentence.split()
    for _ in range(n):
        new_word = random.choice(words)
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, new_word)
    return ' '.join(words)
