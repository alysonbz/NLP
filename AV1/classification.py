import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from atribute_extraction import FeatureExtractor  # Certifique-se de que o FeatureExtractor está no mesmo diretório ou no PYTHONPATH

# Carregar o dataset (ajuste para o caminho correto)
df = pd.read_csv('dataset_final.csv')
df_processed = pd.read_csv('processed_dataset.csv')

# Ajuste de colunas do dataset
corpus_original = df['essay']  # Texto bruto do dataset original
corpus_processed = df_processed['clean_text']  # Texto pré-processado do dataset
labels = df['thematic_coherence']  # Substitua por sua coluna de rótulos no dataset (ajuste conforme necessário)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Função para aplicar stemming
def apply_stemming(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

# Função para aplicar lemmatization
def apply_lemmatization(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Aplicar stemming e lemmatization ao corpus_processed
df_processed['stemmed_text_complete'] = df_processed['clean_text'].apply(apply_stemming)
df_processed['lemmatized_text_complete'] = df_processed['clean_text'].apply(apply_lemmatization)

# Inicializar os extratores de atributos
extractor_original = FeatureExtractor(corpus_original)
extractor_processed = FeatureExtractor(corpus_processed)

# Chamada da função de análise estatística
stats_original = extractor_original.statistical_analysis()
print("Estatísticas do Corpus Original:", stats_original)

stats_processed = extractor_processed.statistical_analysis()
print("Estatísticas do Corpus Processado:", stats_processed)

# Parte A: Comparar os resultados com e sem pré-processamento
results_a = {}

# Comparar a extração de atributos com e sem pré-processamento
for dataset_type, extractor in [("original", extractor_original), ("processed", extractor_processed)]:
    # CountVectorizer
    features_cv, _ = extractor.count_vectorizer()
    X_train, X_test, y_train, y_test = train_test_split(features_cv, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    results_a[f"CountVectorizer ({dataset_type})"] = accuracy

    # TF-IDF
    features_tfidf, _ = extractor.tfidf_vectorizer()
    X_train, X_test, y_train, y_test = train_test_split(features_tfidf, labels, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    results_a[f"TF-IDF ({dataset_type})"] = accuracy

# Exibir resultados da Parte A
print("Parte A: Resultados com e sem pré-processamento")
for key, value in results_a.items():
    print(f"{key}: {value:.4f}")

# Parte B: Comparar formas de extração de atributos com o pré-processamento escolhido
results_b = {}
chosen_preprocessed_corpus = corpus_processed  # Escolha o melhor corpus (processed neste caso)
extractor = FeatureExtractor(chosen_preprocessed_corpus)

# CountVectorizer
features_cv, _ = extractor.count_vectorizer()
X_train, X_test, y_train, y_test = train_test_split(features_cv, labels, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
results_b["CountVectorizer"] = accuracy

# TF-IDF
features_tfidf, _ = extractor.tfidf_vectorizer()
X_train, X_test, y_train, y_test = train_test_split(features_tfidf, labels, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
results_b["TF-IDF"] = accuracy


# Exibir resultados da Parte B
print("\nParte B: Comparação das formas de extração de atributos")
for key, value in results_b.items():
    print(f"{key}: {value:.4f}")

# Parte C: Comparar stemming e lemmatization usando a melhor forma de extração de atributos
best_extraction_method = "TF-IDF"  # Substitua pelo método que obteve melhor desempenho na Parte B
corpus_stemming = df_processed['stemmed_text_complete']  # Coluna com o texto após stemming
corpus_lemmatization = df_processed['lemmatized_text_complete']  # Coluna com o texto após lemmatization
labels = df['thematic_coherence']  # Certifique-se de que a coluna de rótulo está correta

# Stemming
extractor_stemming = FeatureExtractor(corpus_stemming)
features_stemming, _ = extractor_stemming.tfidf_vectorizer()  # Melhor método escolhido
X_train_stemming, X_test_stemming, y_train_stemming, y_test_stemming = train_test_split(
    features_stemming, labels, test_size=0.2, random_state=42
)
clf_stemming = RandomForestClassifier(random_state=42)
clf_stemming.fit(X_train_stemming, y_train_stemming)
pred_stemming = clf_stemming.predict(X_test_stemming)
report_stemming = classification_report(y_test_stemming, pred_stemming, output_dict=True)

# Lemmatization
extractor_lemmatization = FeatureExtractor(corpus_lemmatization)
features_lemmatization, _ = extractor_lemmatization.tfidf_vectorizer()  # Melhor método escolhido
X_train_lemmatization, X_test_lemmatization, y_train_lemmatization, y_test_lemmatization = train_test_split(
    features_lemmatization, labels, test_size=0.2, random_state=42
)
clf_lemmatization = RandomForestClassifier(random_state=42)
clf_lemmatization.fit(X_train_lemmatization, y_train_lemmatization)
pred_lemmatization = clf_lemmatization.predict(X_test_lemmatization)  # Corrigido para X_test_lemmatization
report_lemmatization = classification_report(y_test_lemmatization, pred_lemmatization, output_dict=True)

# Exibir comparação de resultados da Parte C
print("\nParte C: Comparação de Stemming e Lemmatization")
print("\nStemming:")
print(pd.DataFrame(report_stemming).T)

print("\nLemmatization:")
print(pd.DataFrame(report_lemmatization).T)  # Corrigido para report_lemmatization

results_a_df = pd.DataFrame.from_dict(results_a, orient='index', columns=['Accuracy'])
results_b_df = pd.DataFrame.from_dict(results_b, orient='index', columns=['Accuracy'])

stemming_df = pd.DataFrame(report_stemming).T
lemmatization_df = pd.DataFrame(report_lemmatization).T

comparison_df = pd.concat(
    [stemming_df[['precision', 'recall', 'f1-score']].rename(columns=lambda x: f"{x}_Stemming"),
     lemmatization_df[['precision', 'recall', 'f1-score']].rename(columns=lambda x: f"{x}_Lemmatization")],
    axis=1
)

# Exibir resultados
print("\nMatriz de Comparação - Parte A")
print(results_a_df)

print("\nMatriz de Comparação - Parte B")
print(results_b_df)

print("\nMatriz de Comparação - Parte C")
print(comparison_df.to_string())