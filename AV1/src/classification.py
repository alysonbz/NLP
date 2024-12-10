# Import required classes and libraries
import numpy as np
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from AV1.src.atribute_extraction import FeatureExtractor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from AV1.src.utils.grid import param_grid
from AV1.src.utils.defs import text_to_avg_vector

# Preparing processed and raw dataset
df_original = pd.read_csv('portuguese_hate_.csv')
df_processed = pd.read_csv('portuguese_hate_processed_stopwords_manual.csv')
df_processed = df_processed.dropna().reset_index(drop=True)
df_original = df_original.dropna().reset_index(drop=True)
df_original = df_original.loc[df_processed.index].reset_index(drop=True)
print(df_processed['is_hate_speech'].value_counts())
corpus_original = df_original['text']
corpus_processed = df_processed['clean_text']
labels = df_processed['is_hate_speech']

# Feature extraction
extractor_original = FeatureExtractor(corpus_original)
extractor_processed = FeatureExtractor(corpus_processed)

results = {}

# Evaluate Statistical Analysis
for dataset_type, extractor in [("original", extractor_original), ("processed", extractor_processed)]:
    stats = extractor.statistical_analysis()
    print(f"\nStatistical Analysis for {dataset_type} corpus:")
    for key, value in stats.items():
        print(f"{key}: {value}")

# Evaluate Cooccurrence Matrix
for dataset_type, extractor in [("original", extractor_original), ("processed", extractor_processed)]:
    cooccurrence_matrix = extractor.cooccurrence_matrix(window_size=2)
    print(f"\nCooccurrence Matrix for {dataset_type} corpus:")
    print(cooccurrence_matrix)


# Evaluate CountVectorizer
for dataset_type, extractor in [("original", extractor_original), ("processed", extractor_processed)]:
    features_cv, _ = extractor.count_vectorizer()
    X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(features_cv, labels, test_size=0.2, random_state=42, stratify=labels)

    # Model
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train_cv, y_train_cv)
    best_model = grid_search.best_estimator_
    pred = best_model.predict(X_test_cv)
    accuracy = accuracy_score(y_test_cv, pred)
    results[f"CountVectorizer ({dataset_type})"] = accuracy

# Evaluate TF-IDF
for dataset_type, extractor in [("original", extractor_original), ("processed", extractor_processed)]:
    features_tfidf, _ = extractor.tfidf_vectorizer()
    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(features_tfidf, labels, test_size=0.2, random_state=42, stratify=labels)

    # Model
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train_tfidf, y_train_tfidf)
    best_model = grid_search.best_estimator_
    pred = best_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test_tfidf, pred)
    results[f"TF-IDF ({dataset_type})"] = accuracy

# Evaluate WordVec
for dataset_type, extractor in [("original", extractor_original), ("processed", extractor_processed)]:
    word2vec_model = extractor.word2vec(vector_size=100, window=5, min_count=1, epochs=10)

    features_w2v = np.array([text_to_avg_vector(text, word2vec_model) for text in extractor.corpus])

    # Train-test split
    X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(
        features_w2v, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Model
    clf_w2v = RandomForestClassifier(random_state=42)
    grid_search_w2v = GridSearchCV(
        estimator=clf_w2v,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    grid_search_w2v.fit(X_train_w2v, y_train_w2v)
    best_model_w2v = grid_search_w2v.best_estimator_
    pred_w2v = best_model_w2v.predict(X_test_w2v)
    accuracy_w2v = accuracy_score(y_test_w2v, pred_w2v)
    results[f"Word2Vec ({dataset_type})"] = accuracy_w2v

# Print results
print("Results:")
for key, value in results.items():
    print(f"{key}: {value}")





# Item C
smote = SMOTE(random_state=42)
corpus_stemming = df_processed['stemmed_text_complete']
corpus_lemmatization = df_processed['lemmatized_text_complete']

extractor_stemming = FeatureExtractor(corpus_stemming)
word2vec_model_stemming = extractor_stemming.word2vec(vector_size=100, window=5, min_count=1, epochs=10)
features_stemming = np.array([text_to_avg_vector(text, word2vec_model_stemming) for text in corpus_stemming])

# Aplicação do SMOTE para balanceamento
features_stemming_resampled, labels_stemming_resampled = smote.fit_resample(features_stemming, labels)

# Divisão de treino e teste para stemming
X_train_stemming, X_test_stemming, y_train_stemming, y_test_stemming = train_test_split(
    features_stemming_resampled, labels_stemming_resampled, test_size=0.2, random_state=42, stratify=labels_stemming_resampled
)

# Treinamento com XGBoost para stemming
clf_stemming = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
clf_stemming.fit(X_train_stemming, y_train_stemming)
pred_stemming = clf_stemming.predict(X_test_stemming)
report_stemming = classification_report(y_test_stemming, pred_stemming, output_dict=True)

# Lemmatization com Word2Vec
extractor_lemmatization = FeatureExtractor(corpus_lemmatization)
word2vec_model_lemmatization = extractor_lemmatization.word2vec(vector_size=100, window=5, min_count=1, epochs=10)
features_lemmatization = np.array([text_to_avg_vector(text, word2vec_model_lemmatization) for text
                                   in corpus_lemmatization])

# Aplicação do SMOTE para balanceamento
features_lemmatization_resampled, labels_lemmatization_resampled = smote.fit_resample(features_lemmatization, labels)

# Divisão de treino e teste para lemmatization
X_train_lemmatization, X_test_lemmatization, y_train_lemmatization, y_test_lemmatization = train_test_split(
    features_lemmatization_resampled, labels_lemmatization_resampled, test_size=0.2, random_state=42, stratify=
    labels_lemmatization_resampled
)

# Treinamento com XGBoost para lemmatization
clf_lemmatization = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
clf_lemmatization.fit(X_train_lemmatization, y_train_lemmatization)
pred_lemmatization = clf_lemmatization.predict(X_test_lemmatization)
report_lemmatization = classification_report(y_test_lemmatization, pred_lemmatization, output_dict=True)

# Resultados
print("\nComparison of Stemming and Lemmatization using Word2Vec with SMOTE:")
print("\nStemming:")
print(pd.DataFrame(report_stemming).T)

print("\nLemmatization:")
print(pd.DataFrame(report_lemmatization).T)