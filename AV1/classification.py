import pandas as pd
from preprocessing import preprocess_text, remove_stopwords, stem_text, lemmatize_text
from vectorizer import count_vectorizer, tfidf_vectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np


# Função para carregar e pré-processar os dados
def load_and_preprocess_data(filepath, use_preprocessing=True, use_stemming=False, use_lemmatization=False,
                             sample_size=None):
    df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    df_true = df[['message_true']].copy()
    df_true = df_true.dropna()
    df_true['type'] = 'true'
    df_true['message'] = df_true['message_true']
    df = pd.concat([df[['message', 'type']], df_true[['message', 'type']]], ignore_index=True)

    df['message'] = df['message'].fillna('')

    if use_preprocessing:
        df['message_norm'] = df['message'].apply(preprocess_text)
        df['message_norm'] = df['message_norm'].apply(remove_stopwords)

        if use_stemming:
            df['message_norm'] = df['message_norm'].apply(stem_text)
        elif use_lemmatization:
            df['message_norm'] = df['message_norm'].apply(lemmatize_text)
    else:
        df['message_norm'] = df['message']  # Sem pré-processamento

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


# (a) Comparação do TF-IDF com e sem pré-processamento
def compare_tfidf_preprocessing(filepath, sample_size=None):
    results = []

    # Com pré-processamento
    df_preprocessed = load_and_preprocess_data(filepath, use_preprocessing=True, sample_size=sample_size)
    X_train, X_test, y_train, y_test = train_test_split_with_class_distribution(df_preprocessed)
    X_train_vectorized, vectorizer = tfidf_vectorizer(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy, precision, recall = train_and_evaluate(X_train_vectorized, X_test_vectorized, y_train, y_test)
    results.append({
        "method": "TF-IDF com pré-processamento",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })

    # Sem pré-processamento
    df_no_preprocessing = load_and_preprocess_data(filepath, use_preprocessing=False, sample_size=sample_size)
    X_train, X_test, y_train, y_test = train_test_split_with_class_distribution(df_no_preprocessing)
    X_train_vectorized, vectorizer = tfidf_vectorizer(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy, precision, recall = train_and_evaluate(X_train_vectorized, X_test_vectorized, y_train, y_test)
    results.append({
        "method": "TF-IDF sem pré-processamento",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })

    return results


# (b) Comparação entre CountVectorizer e TF-IDF com o melhor pré-processamento do item (a)
def compare_vectorization_methods(filepath, sample_size=None, use_stemming=False, use_lemmatization=False):
    results = []

    # Com CountVectorizer
    df_preprocessed = load_and_preprocess_data(filepath, use_stemming=use_stemming, use_lemmatization=use_lemmatization,
                                               sample_size=sample_size)
    X_train, X_test, y_train, y_test = train_test_split_with_class_distribution(df_preprocessed)
    X_train_vectorized, vectorizer = count_vectorizer(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy, precision, recall = train_and_evaluate(X_train_vectorized, X_test_vectorized, y_train, y_test)
    results.append({
        "method": "CountVectorizer com melhor pré-processamento",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })

    # Com TF-IDF
    X_train_vectorized, vectorizer = tfidf_vectorizer(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy, precision, recall = train_and_evaluate(X_train_vectorized, X_test_vectorized, y_train, y_test)
    results.append({
        "method": "TF-IDF com melhor pré-processamento",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })

    return results


# (c) Comparação entre Lematização e Stemming usando a melhor vetorização do item (b)
def compare_stemming_lemmatization(filepath, sample_size=None, vectorization_method="TF-IDF"):
    results = []

    # Com Stemming
    df_stemming = load_and_preprocess_data(filepath, use_stemming=True, sample_size=sample_size)
    X_train, X_test, y_train, y_test = train_test_split_with_class_distribution(df_stemming)

    if vectorization_method == "TF-IDF":
        X_train_vectorized, vectorizer = tfidf_vectorizer(X_train)
    else:
        X_train_vectorized, vectorizer = count_vectorizer(X_train)

    X_test_vectorized = vectorizer.transform(X_test)
    accuracy, precision, recall = train_and_evaluate(X_train_vectorized, X_test_vectorized, y_train, y_test)
    results.append({
        "method": f"{vectorization_method} com Stemming",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })

    # Com Lematização
    df_lemmatization = load_and_preprocess_data(filepath, use_lemmatization=True, sample_size=sample_size)
    X_train, X_test, y_train, y_test = train_test_split_with_class_distribution(df_lemmatization)

    if vectorization_method == "TF-IDF":
        X_train_vectorized, vectorizer = tfidf_vectorizer(X_train)
    else:
        X_train_vectorized, vectorizer = count_vectorizer(X_train)

    X_test_vectorized = vectorizer.transform(X_test)
    accuracy, precision, recall = train_and_evaluate(X_train_vectorized, X_test_vectorized, y_train, y_test)
    results.append({
        "method": f"{vectorization_method} com Lematização",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })

    return results


# Caminho do arquivo
filepath = "C:\\Users\\mateu\\Downloads\\complete_dataset\\complete_dataset.csv"

# Defina o tamanho da amostra para acelerar a execução
sample_size = 200  # Ajuste conforme necessário

# (a) Comparando TF-IDF com e sem pré-processamento
results_a = compare_tfidf_preprocessing(filepath, sample_size=sample_size)
print("Resultados da comparação TF-IDF com e sem pré-processamento:")
for result in results_a:
    print(f"Method: {result['method']}")
    print(f"  Accuracy: {result['accuracy']:.2f}")
    print(f"  Precision: {result['precision']:.2f}")
    print(f"  Recall: {result['recall']:.2f}\n")

# Escolha a melhor abordagem de pré-processamento do item (a)
use_stemming = True  # ou False, dependendo dos resultados da comparação

# (b) Comparando CountVectorizer vs TF-IDF com o melhor pré-processamento
results_b = compare_vectorization_methods(filepath, sample_size=sample_size, use_stemming=use_stemming)
print("Resultados da comparação entre CountVectorizer e TF-IDF:")
for result in results_b:
    print(f"Method: {result['method']}")
    print(f"  Accuracy: {result['accuracy']:.2f}")
    print(f"  Precision: {result['precision']:.2f}")
    print(f"  Recall: {result['recall']:.2f}\n")

# Escolha a melhor vetorização do item (b)
best_vectorization = "TF-IDF"  # ou "CountVectorizer", dependendo dos resultados da comparação

# (c) Comparando Lematização vs Stemming usando a melhor vetorização
results_c = compare_stemming_lemmatization(filepath, sample_size=sample_size, vectorization_method=best_vectorization)
print("Resultados da comparação entre Lematização e Stemming:")
for result in results_c:
    print(f"Method: {result['method']}")
    print(f"  Accuracy: {result['accuracy']:.2f}")
    print(f"  Precision: {result['precision']:.2f}")
    print(f"  Recall: {result['recall']:.2f}\n")

# Resultados simulados, substitua pelos resultados reais obtidos
results_a = [
    {"method": "TF-IDF com pré-processamento", "accuracy": 0.82, "precision": 0.87, "recall": 0.82},
    {"method": "TF-IDF sem pré-processamento", "accuracy": 0.79, "precision": 0.85, "recall": 0.79}
]

results_b = [
    {"method": "CountVectorizer", "accuracy": 0.80, "precision": 0.86, "recall": 0.80},
    {"method": "TF-IDF", "accuracy": 0.82, "precision": 0.87, "recall": 0.82}
]

results_c = [
    {"method": "TF-IDF com Stemming", "accuracy": 0.82, "precision": 0.87, "recall": 0.82},
    {"method": "TF-IDF com Lematização", "accuracy": 0.82, "precision": 0.87, "recall": 0.82}
]

# Função para criar gráficos de barras
# Função para criar gráficos de barras com numeração
def plot_comparison(results, title):
    methods = [result['method'] for result in results]
    accuracy = [result['accuracy'] for result in results]
    precision = [result['precision'] for result in results]
    recall = [result['recall'] for result in results]

    x = np.arange(len(methods))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, accuracy, width, label='Accuracy')
    rects2 = ax.bar(x, precision, width, label='Precision')
    rects3 = ax.bar(x + width, recall, width, label='Recall')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Method')
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    # Função para adicionar rótulos acima das barras
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Adiciona os rótulos nas barras
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    fig.tight_layout()
    plt.show()

# Plotando os gráficos para cada comparação
plot_comparison(results_a, 'TF-IDF com vs sem pré-processamento')
plot_comparison(results_b, 'CountVectorizer vs TF-IDF')
plot_comparison(results_c, 'Lematização vs Stemming')
