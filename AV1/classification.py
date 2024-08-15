import pandas as pd
from preprocessing import preprocess_text, remove_stopwords, stem_text, lemmatize_text
from vectorizer import count_vectorizer, tfidf_vectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.utils import shuffle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Função para verificar a distribuição das classes no conjunto completo
def check_class_distribution(df):
    class_distribution = df['type'].value_counts()
    print("Distribuição das classes no conjunto completo:")
    print(class_distribution)

# Função para realizar análise de sentimento
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores['compound']

# Função para amostrar os dados
def sample_data(df, sample_size=None):
    if sample_size:
        return df.sample(n=sample_size, random_state=42)
    return df

# Função para carregar e pré-processar os dados
def load_and_preprocess_data(filepath, use_stemming=True, use_lemmatization=False, sample_size=None):
    df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')

    # Aplicando a amostragem
    df = sample_data(df, sample_size)

    print("Colunas disponíveis:", df.columns.tolist())

    df_true = df[['message_true']].copy()
    df_true = df_true.dropna()
    df_true['type'] = 'true'
    df_true['message'] = df_true['message_true']
    df = pd.concat([df[['message', 'type']], df_true[['message', 'type']]], ignore_index=True)

    df['message'] = df['message'].fillna('')
    df['message_norm'] = df['message'].apply(preprocess_text)
    df['message_norm'] = df['message_norm'].apply(remove_stopwords)

    if use_stemming:
        df['message_norm'] = df['message_norm'].apply(stem_text)
    elif use_lemmatization:
        df['message_norm'] = df['message_norm'].apply(lemmatize_text)

    df['sentiment'] = df['message'].apply(analyze_sentiment)

    return df

# Função para dividir e garantir a presença de todas as classes
def train_test_split_with_class_distribution(df):
    df = shuffle(df, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(df['message_norm'], df['type'], test_size=0.2,
                                                        stratify=df['type'], random_state=42)
    return X_train, X_test, y_train, y_test

# Função para treinar e avaliar o modelo
def train_and_evaluate(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    return accuracy, precision, recall

# Função para comparar as técnicas de pré-processamento e vetorização
def compare_preprocessing_techniques(filepath, sample_size=None):
    results = {}

    preprocessing_options = [
        {"use_stemming": True, "use_lemmatization": False, "method": "Stemming + CountVectorizer"},
        {"use_stemming": True, "use_lemmatization": False, "method": "Stemming + TF-IDF"},
        {"use_stemming": False, "use_lemmatization": True, "method": "Lemmatization + CountVectorizer"},
        {"use_stemming": False, "use_lemmatization": True, "method": "Lemmatization + TF-IDF"}
    ]

    for option in preprocessing_options:
        df_preprocessed = load_and_preprocess_data(filepath,
                                                   use_stemming=option["use_stemming"],
                                                   use_lemmatization=option["use_lemmatization"],
                                                   sample_size=sample_size)
        X_train, X_test, y_train, y_test = train_test_split_with_class_distribution(df_preprocessed)

        if "CountVectorizer" in option["method"]:
            X_train_vectorized, vectorizer = count_vectorizer(X_train)
            X_test_vectorized = vectorizer.transform(X_test)
        elif "TF-IDF" in option["method"]:
            X_train_vectorized, vectorizer = tfidf_vectorizer(X_train)
            X_test_vectorized = vectorizer.transform(X_test)

        accuracy, precision, recall = train_and_evaluate(X_train_vectorized, X_test_vectorized, y_train, y_test)
        results[option["method"]] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }

    return results

# Caminho do arquivo
filepath = "C:\\Users\\mateu\\Downloads\\complete_dataset\\complete_dataset.csv"

# Defina o tamanho da amostra para acelerar a execução
sample_size = 200  # Ajuste conforme necessário

# Executando a comparação
results = compare_preprocessing_techniques(filepath, sample_size=sample_size)

# Exibindo os resultados
for method, metrics in results.items():
    print(f"Method: {method}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")

# Análise de sentimento no dataset completo
df_full = load_and_preprocess_data(filepath, sample_size=sample_size)
df_full['sentiment'] = df_full['message'].apply(analyze_sentiment)
print(df_full[['message', 'sentiment', 'type']].head())
