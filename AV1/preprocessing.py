import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Define as stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Inicializa o stemmer e o lemmatizer
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Função de pré-processamento básico
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove caracteres não alfanuméricos
    text = re.sub(r'\s+', ' ', text)  # Remove espaços múltiplos
    text = text.lower().strip()  # Converte para minúsculas e remove espaços nas extremidades
    return text

# Função para remover stopwords
def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Função para aplicar stemming
def stem_text(text):
    words = text.split()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Função para aplicar lemmatization
def lemmatize_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Texto de exemplo
example_text = "O rato roeu a roupa do rei de Roma. Além disso, havia muitas dúvidas sobre a qualidade da roupa."

# Aplicando as funções
preprocessed_text = preprocess_text(example_text)
no_stopwords_text = remove_stopwords(preprocessed_text)
stemmed_text = stem_text(no_stopwords_text)
lemmatized_text = lemmatize_text(no_stopwords_text)

# Imprimindo os resultados
print("Texto original:")
print(example_text)
print("\nTexto após pré-processamento básico:")
print(preprocessed_text)
print("\nTexto após remoção de stopwords:")
print(no_stopwords_text)
print("\nTexto após aplicação de stemming:")
print(stemmed_text)
print("\nTexto após aplicação de lematização:")
print(lemmatized_text)
