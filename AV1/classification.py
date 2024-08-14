import pandas as pd
from preprocessing import preprocess_text, remove_stopwords, stem_text, lemmatize_text
from vectorizer import count_vectorizer, tfidf_vectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


# Função para carregar e pré-processar os dados
def load_and_preprocess_data(filepath, use_stemming=True, use_lemmatization=False, sample_size=1000):
    print(f"Carregando os dados de {filepath} com amostra de tamanho {sample_size}...")
    df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')

    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    df_true = df[['message_true']].copy()
    df_true = df_true.dropna()
    df_true['type'] = 'true'
    df_true['message'] = df_true['message_true']
    df = pd.concat([df[['message', 'type']], df_true[['message', 'type']]], ignore_index=True)

    df['message'] = df['message'].fillna('')
    print("Pré-processando mensagens...")
    df['message_norm'] = df['message'].apply(preprocess_text)
    df['message_norm'] = df['message_norm'].apply(remove_stopwords)

    if use_stemming:
        print("Aplicando stemming...")
        df['message_norm'] = df['message_norm'].apply(stem_text)
    elif use_lemmatization:
        print("Aplicando lemmatization...")
        df['message_norm'] = df['message_norm'].apply(lemmatize_text)

    return df


# Função para dividir e garantir a presença de todas as classes
def train_test_split_with_class_distribution(df):
    print("Dividindo os dados em conjuntos de treinamento e teste...")
    df = shuffle(df, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(df['message_norm'], df['type'], test_size=0.2,
                                                        stratify=df['type'], random_state=42)
    return X_train, X_test, y_train, y_test


# Função para treinar e avaliar um modelo
def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("Treinando e avaliando o modelo...")
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


# Parte a) Comparar TF-IDF com e sem pré-processamento
def compare_with_without_preprocessing(filepath):
    print("Comparando TF-IDF com e sem pré-processamento...")
    df_raw = load_and_preprocess_data(filepath, use_stemming=False, use_lemmatization=False)
    df_preprocessed = load_and_preprocess_data(filepath, use_stemming=True, use_lemmatization=False)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split_with_class_distribution(df_raw)
    X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split_with_class_distribution(df_preprocessed)

    # Vetorização TF-IDF sem pré-processamento
    print("Vetorização TF-IDF sem pré-processamento...")
    X_train_tfidf_raw, vectorizer_raw = tfidf_vectorizer(X_train_raw)
    X_test_tfidf_raw = vectorizer_raw.transform(X_test_raw)
    accuracy_raw = train_and_evaluate(X_train_tfidf_raw, X_test_tfidf_raw, y_train_raw, y_test_raw)

    # Vetorização TF-IDF com pré-processamento
    print("Vetorização TF-IDF com pré-processamento...")
    X_train_tfidf_pre, vectorizer_pre = tfidf_vectorizer(X_train_pre)
    X_test_tfidf_pre = vectorizer_pre.transform(X_test_pre)
    accuracy_pre = train_and_evaluate(X_train_tfidf_pre, X_test_tfidf_pre, y_train_pre, y_test_pre)

    return {"TF-IDF sem pré-processamento": accuracy_raw, "TF-IDF com pré-processamento": accuracy_pre}


# Parte b) Comparar CountVectorizer vs TF-IDF com pré-processamento
def compare_vectorizers(filepath):
    print("Comparando CountVectorizer vs TF-IDF com pré-processamento...")
    df = load_and_preprocess_data(filepath, use_stemming=True, use_lemmatization=False)
    X_train, X_test, y_train, y_test = train_test_split_with_class_distribution(df)

    # CountVectorizer
    print("Vetorização CountVectorizer...")
    X_train_count, vectorizer_count = count_vectorizer(X_train)
    X_test_count = vectorizer_count.transform(X_test)
    accuracy_count = train_and_evaluate(X_train_count, X_test_count, y_train, y_test)

    # TF-IDF
    print("Vetorização TF-IDF...")
    X_train_tfidf, vectorizer_tfidf = tfidf_vectorizer(X_train)
    X_test_tfidf = vectorizer_tfidf.transform(X_test)
    accuracy_tfidf = train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test)

    return {"CountVectorizer": accuracy_count, "TF-IDF": accuracy_tfidf}


# Parte c) Comparar stemming vs lemmatization com a melhor vetorização do item b)
def compare_stemming_lemmatization(filepath):
    print("Comparando Stemming vs Lemmatization com a melhor vetorização...")
    df_stemming = load_and_preprocess_data(filepath, use_stemming=True, use_lemmatization=False)
    df_lemmatization = load_and_preprocess_data(filepath, use_stemming=False, use_lemmatization=True)

    X_train_stem, X_test_stem, y_train_stem, y_test_stem = train_test_split_with_class_distribution(df_stemming)
    X_train_lem, X_test_lem, y_train_lem, y_test_lem = train_test_split_with_class_distribution(df_lemmatization)

    # Usando TF-IDF (assumindo que foi o melhor vetor)
    print("Usando TF-IDF para comparar Stemming e Lemmatization...")
    X_train_tfidf_stem, vectorizer_stem = tfidf_vectorizer(X_train_stem)
    X_test_tfidf_stem = vectorizer_stem.transform(X_test_stem)
    accuracy_stem = train_and_evaluate(X_train_tfidf_stem, X_test_tfidf_stem, y_train_stem, y_test_stem)

    X_train_tfidf_lem, vectorizer_lem = tfidf_vectorizer(X_train_lem)
    X_test_tfidf_lem = vectorizer_lem.transform(X_test_lem)
    accuracy_lem = train_and_evaluate(X_train_tfidf_lem, X_test_tfidf_lem, y_train_lem, y_test_lem)

    return {"Stemming": accuracy_stem, "Lemmatization": accuracy_lem}


# Caminho do arquivo
filepath = "C:\\Users\\mateu\\Downloads\\complete_dataset\\complete_dataset.csv"

# Executando as comparações com uma amostra de 1000 dados
sample_size = 200  # Ajuste o tamanho da amostra conforme necessário

# Parte a)
results_a = compare_with_without_preprocessing(filepath)
print("Resultados - TF-IDF com e sem pré-processamento:", results_a)

# Parte b)
results_b = compare_vectorizers(filepath)
print("Resultados - CountVectorizer vs TF-IDF:", results_b)

# Parte c)
results_c = compare_stemming_lemmatization(filepath)
print("Resultados - Stemming vs Lemmatization:", results_c)
