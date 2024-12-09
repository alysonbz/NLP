# Import required classes and libraries
from AV1.atribute_extraction import FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from xgboost import XGBClassifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Preparing processed and raw dataset
df_original = pd.read_csv('../AV1/dataset/twitter_sentiment.csv')
df_processed = pd.read_csv('../AV1/dataset/twitter_sentiment_processed.csv')
df_processed = df_processed.dropna().reset_index(drop=True)
df_original = df_original.dropna().reset_index(drop=True)
df_original = df_original.loc[df_processed.index].reset_index(drop=True)
df_processed = df_processed.reset_index(drop=True)

corpus_original = df_original['tweet_text']
corpus_processed = df_processed['clean_text']
labels = df_original['sentiment']

# Feature extraction
extractor_original = FeatureExtractor(corpus_original)
extractor_processed = FeatureExtractor(corpus_processed)

results = {}
# Avaliando CountVectorizer
for dataset_type, extractor in [("original", extractor_original), ("processed", extractor_processed)]:
    features_cv, _ = extractor.count_vectorizer()
    X_train, X_test, y_train, y_test = train_test_split(features_cv, labels, test_size=0.2, random_state=42)
    clf = XGBClassifier(random_state=42, n_estimators=200, max_depth=6)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    results[f"CountVectorizer ({dataset_type})"] = accuracy

# Avaliando TF-IDF
for dataset_type, extractor in [("original", extractor_original), ("processed", extractor_processed)]:
    features_tfidf, _ = extractor.tfidf_vectorizer()
    X_train, X_test, y_train, y_test = train_test_split(features_tfidf, labels, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    results[f"TF-IDF ({dataset_type})"] = accuracy

# Avaliação de Word2Vec
for dataset_type, extractor in [("original", extractor_original), ("processed", extractor_processed)]:
    w2v_model = extractor.word2vec()
    def text_to_vector(text):
        words = text.split()
        word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:  #
            return np.zeros(w2v_model.vector_size)


    features_w2v = np.array([text_to_vector(text) for text in extractor.corpus])

    X_train, X_test, y_train, y_test = train_test_split(features_w2v, labels, test_size=0.2, random_state=42)

    clf = XGBClassifier(random_state=42, n_estimators=200, max_depth=6)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    results[f"Word2Vec ({dataset_type})"] = accuracy

# Avaliando Statistical Analysis
for dataset_type, extractor in [("original", extractor_original), ("processed", extractor_processed)]:
    stats = extractor.statistical_analysis()
    features_stats = np.array([[stats["mean_length"], stats["max_length"], stats["min_length"],
                                 stats["total_tokens"], stats["var_tokens"]]] * len(labels))  # Replicando os stats para cada amostra

    X_train, X_test, y_train, y_test = train_test_split(features_stats, labels, test_size=0.2, random_state=42)

    clf = XGBClassifier(random_state=42, n_estimators=200, max_depth=6)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    results[f"Statistical Analysis ({dataset_type})"] = accuracy

# Avaliando Cooccurrence Matrix
for dataset_type, extractor in [("original", extractor_original), ("processed", extractor_processed)]:
    cooccurrence_matrix = extractor.cooccurrence_matrix()

    # Convertendo a matriz de coocorrência para características por amostra
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(extractor.corpus)
    vocab = vectorizer.get_feature_names_out()

    features_cooccurrence = []
    for text in extractor.corpus:
        words = text.split()
        vector = cooccurrence_matrix.loc[words, words].sum().sum() if all(word in vocab for word in words) else 0
        features_cooccurrence.append(vector)

    features_cooccurrence = np.array(features_cooccurrence).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(features_cooccurrence, labels, test_size=0.2, random_state=42)

    clf = XGBClassifier(random_state=42, n_estimators=200, max_depth=6)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    results[f"Cooccurrence Matrix ({dataset_type})"] = accuracy

print("Results:")
for key, value in results.items():
    print(f"{key}: {value}")

# Resolução Item C
best_extraction_method = "Word2Vec"
corpus_stemming = df_processed['stemmed_text']
corpus_lemmatization = df_processed['lemmatized_text']
labels = df_processed['sentiment']
extractor_stemming = FeatureExtractor(corpus_stemming)
features_stemming, _ = extractor_stemming.tfidf_vectorizer()

X_train_stemming, X_test_stemming, y_train_stemming, y_test_stemming = train_test_split(
    features_stemming, labels, test_size=0.2, random_state=42
)

clf_stemming = XGBClassifier(random_state=42, n_estimators=200, max_depth=6)
clf_stemming.fit(X_train_stemming, y_train_stemming)
pred_stemming = clf_stemming.predict(X_test_stemming)
report_stemming = classification_report(y_test_stemming, pred_stemming, output_dict=True)

# Avaliação com textos Lemmatization
extractor_lemmatization = FeatureExtractor(corpus_lemmatization)
features_lemmatization, _ = extractor_lemmatization.tfidf_vectorizer()

X_train_lemmatization, X_test_lemmatization, y_train_lemmatization, y_test_lemmatization = train_test_split(
    features_lemmatization, labels, test_size=0.2, random_state=42
)

clf_lemmatization = XGBClassifier(random_state=42, n_estimators=200, max_depth=6)
clf_lemmatization.fit(X_train_lemmatization, y_train_lemmatization)
pred_lemmatization = clf_lemmatization.predict(X_test_lemmatization)
report_lemmatization = classification_report(y_test_lemmatization, pred_lemmatization, output_dict=True)

# Comparação de resultados
print("\nComparison of Stemming and Lemmatization:")
print("\nStemming:")
print(pd.DataFrame(report_stemming).T)

print("\nLemmatization:")
print(pd.DataFrame(report_lemmatization).T)
