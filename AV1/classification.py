import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import nltk

# Baixar recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Função para pré-processar o texto
def preprocess_text(text, apply_lemma=False):
    if not isinstance(text, str):
        raise ValueError("O texto deve ser uma string.")
    text = text.lower()  # Converter para minúsculas
    text = re.sub(r'\s+', ' ', text)  # Remover espaços em branco extras
    text = re.sub(r'\W+', ' ', text)  # Remover caracteres não alfanuméricos

    # Tokenização
    tokens = word_tokenize(text)
    # Remover stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lematização ou Stemming
    lemmatizer = WordNetLemmatizer()
    if apply_lemma:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    else:
        # Exemplo simples de stemming
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)  # Retornar o texto limpo


# Função para treinar e avaliar o classificador
def train_and_evaluate(X_train, X_test, y_train, y_test):
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Função para comparar resultados com e sem pré-processamento usando TF-IDF
def comparison_with_and_without_preprocessing(texts, labels):
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Vetorização TF-IDF sem pré-processamento
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    accuracy_no_preprocessing = train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test)
    print(f"Sem pré-processamento (TF-IDF): Acurácia: {accuracy_no_preprocessing:.4f}")

    # Vetorização TF-IDF com pré-processamento
    preprocessed_train = [preprocess_text(text, apply_lemma=True) for text in X_train]
    preprocessed_test = [preprocess_text(text, apply_lemma=True) for text in X_test]

    # Diagnóstico: Mostrar alguns exemplos de textos antes e depois do pré-processamento
    print("\nExemplos de texto antes e depois do pré-processamento:")
    for original, preprocessed in zip(X_train[:5], preprocessed_train[:5]):
        print(f"Original: {original}\nPré-processado: {preprocessed}\n")

    vectorizer = TfidfVectorizer()
    X_train_tfidf_preprocessed = vectorizer.fit_transform(preprocessed_train)
    X_test_tfidf_preprocessed = vectorizer.transform(preprocessed_test)

    accuracy_with_preprocessing = train_and_evaluate(X_train_tfidf_preprocessed, X_test_tfidf_preprocessed, y_train,
                                                     y_test)
    print(f"Com pré-processamento (TF-IDF): Acurácia: {accuracy_with_preprocessing:.4f}")

    # Comparar CountVectorizer e TF-IDF com pré-processamento
    vectorizer_cv = CountVectorizer()
    X_train_cv = vectorizer_cv.fit_transform(preprocessed_train)
    X_test_cv = vectorizer_cv.transform(preprocessed_test)

    accuracy_cv = train_and_evaluate(X_train_cv, X_test_cv, y_train, y_test)
    print(f"CountVectorizer com pré-processamento: Acurácia: {accuracy_cv:.4f}")


# Carregar os datasets
fake_news_path = "C:/Users/bianc/OneDrive/Documentos/NLP/AV1/News_fake.csv"
not_fake_news_path = "C:/Users/bianc/OneDrive/Documentos/NLP/AV1/News_notFake.csv"

df_fake = pd.read_csv(fake_news_path)
df_not_fake = pd.read_csv(not_fake_news_path)

# Verifique se os datasets foram carregados corretamente
if df_fake.empty or df_not_fake.empty:
    raise ValueError("Um ou ambos os datasets estão vazios. Verifique o caminho dos arquivos.")

# Unir os datasets e criar rótulos
df_fake['label'] = 1
df_not_fake['label'] = 0
df = pd.concat([df_fake, df_not_fake])

# Verificar se a coluna 'title' está presente
if 'title' not in df.columns:
    raise ValueError("A coluna 'title' não foi encontrada no dataset.")

# Separar os textos e os rótulos
texts = df['title'].values
labels = df['label'].values

# Comparar com e sem pré-processamento usando TF-IDF
comparison_with_and_without_preprocessing(texts, labels)
