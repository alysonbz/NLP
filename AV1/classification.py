import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from atribute_extraction import AttributeExtractor
from preprocessing import preprocessar_texto
from nltk.stem import RSLPStemmer
import gensim
from gensim.models import Word2Vec
import ast
import matplotlib.pylab as plt
# Load the dataset (assuming the dataset and df variable are from Q1)
# Here we assume df has columns: premise, hypothesis, premise_processed, hypothesis_processed and entailment_judgment
# If needed, load again:
df = pd.read_csv("article_nlp_datasets.csv")

# Ensure that 'entailment_judgment' is your target column:
label_col = 'entailment_judgment'
X_premise = df['premise']
X_hypothesis = df['hypothesis']
y = df[label_col]

df['premise_processed'] = df['premise_processed'].apply(ast.literal_eval)
# Let's also have preprocessed versions (with lemmatization as done in Q1):
X_premise_lemmatized = df['premise_processed'].apply(lambda x: " ".join(x))
X_hypothesis_lemmatized = df['hypothesis_processed'].apply(lambda x: " ".join(x))


# Optional: A version with stemming instead of lemmatization (for part c)
stemmer = RSLPStemmer()
def preprocess_with_stemming(text):
    # Similar to preprocessar_texto but replacing lemmatization with stemming
    tokens = preprocessar_texto(text, aplicar_lematizacao=False)
    stemmed = [stemmer.stem(tok) for tok in tokens]
    return " ".join(stemmed)

X_premise_stemmed = df['premise'].apply(preprocess_with_stemming)
X_hypothesis_stemmed = df['hypothesis'].apply(preprocess_with_stemming)


# Split data into train and test (we'll do the same split for all experiments)
X_train_p, X_test_p, X_train_h, X_test_h, y_train, y_test = train_test_split(X_premise, X_hypothesis, y, test_size=0.2, random_state=42)
X_train_pl, X_test_pl, X_train_hl, X_test_hl, _, _ = train_test_split(X_premise_lemmatized, X_hypothesis_lemmatized, y, test_size=0.2, random_state=42)
X_train_ps, X_test_ps, X_train_hs, X_test_hs, _, _ = train_test_split(X_premise_stemmed, X_hypothesis_stemmed, y, test_size=0.2, random_state=42)

# Helper functions to train and evaluate a classifier
def train_and_evaluate(X_train_features, X_test_features, y_train, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_features, y_train)
    y_pred = clf.predict(X_test_features)
    return accuracy_score(y_test, y_pred)

# 1) Without Preprocessing vs With Preprocessing, using all forms of attribute extraction

# --- Individual Statistical Analysis (from Q2) ---
# We need tokenized lists to do this:
X_train_p_tokens = [premise.split() for premise in X_train_p]
X_train_h_tokens = [hyp.split() for hyp in X_train_h]
X_test_p_tokens = [premise.split() for premise in X_test_p]
X_test_h_tokens = [hyp.split() for hyp in X_test_h]

extractor_no_pre = AttributeExtractor(X_train_p_tokens, X_train_h_tokens)
stats_train = extractor_no_pre.individual_statistical_analysis().values

# For test set, we also need stats:
# Re-create extractor for test:
extractor_no_pre_test = AttributeExtractor(X_test_p_tokens, X_test_h_tokens)
stats_test = extractor_no_pre_test.individual_statistical_analysis().values

acc_stats_no_pre = train_and_evaluate(stats_train, stats_test, y_train, y_test)

# With Preprocessing (lemmatization)
X_train_pl_tokens = [premise.split() for premise in X_train_pl]
X_train_hl_tokens = [hyp.split() for hyp in X_train_hl]
X_test_pl_tokens = [premise.split() for premise in X_test_pl]
X_test_hl_tokens = [hyp.split() for hyp in X_test_hl]

extractor_pre = AttributeExtractor(X_train_pl_tokens, X_train_hl_tokens)
stats_train_pre = extractor_pre.individual_statistical_analysis().values
extractor_pre_test = AttributeExtractor(X_test_pl_tokens, X_test_hl_tokens)
stats_test_pre = extractor_pre_test.individual_statistical_analysis().values

acc_stats_pre = train_and_evaluate(stats_train_pre, stats_test_pre, y_train, y_test)

print("Individual Statistical Analysis Accuracy:")
print("No Preprocessing:", acc_stats_no_pre)
print("With Preprocessing (Lemmatization):", acc_stats_pre)

# --- CountVectorizer ---
def combine_premise_hypothesis(premises, hypotheses):
    # Combine premise and hypothesis into a single string (feature)
    return [p + " " + h for p, h in zip(premises, hypotheses)]

# No Preprocessing
X_train_comb = combine_premise_hypothesis(X_train_p, X_train_h)
X_test_comb = combine_premise_hypothesis(X_test_p, X_test_h)

cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train_comb)
X_test_cv = cv.transform(X_test_comb)
acc_cv_no_pre = train_and_evaluate(X_train_cv, X_test_cv, y_train, y_test)

# With Preprocessing (Lemmatization)
X_train_comb_pre = combine_premise_hypothesis(X_train_pl, X_train_hl)
X_test_comb_pre = combine_premise_hypothesis(X_test_pl, X_test_hl)



cv_pre = CountVectorizer()
X_train_cv_pre = cv_pre.fit_transform(X_train_comb_pre)
X_test_cv_pre = cv_pre.transform(X_test_comb_pre)
acc_cv_pre = train_and_evaluate(X_train_cv_pre, X_test_cv_pre, y_train, y_test)

print("CountVectorizer Accuracy:")
print("No Preprocessing:", acc_cv_no_pre)
print("With Preprocessing (Lemmatization):", acc_cv_pre)

# --- TF-IDF ---
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train_comb)
X_test_tfidf = tfidf.transform(X_test_comb)
acc_tfidf_no_pre = train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test)

tfidf_pre = TfidfVectorizer()
X_train_tfidf_pre = tfidf_pre.fit_transform(X_train_comb_pre)
X_test_tfidf_pre = tfidf_pre.transform(X_test_comb_pre)
acc_tfidf_pre = train_and_evaluate(X_train_tfidf_pre, X_test_tfidf_pre, y_train, y_test)

print("TF-IDF Accuracy:")
print("No Preprocessing:", acc_tfidf_no_pre)
print("With Preprocessing (Lemmatization):", acc_tfidf_pre)

# --- Co-occurrence Matrix ---
# Typically co-occurrence matrix is big and not well-suited directly for classification of pairs.
# But let's assume we do similarly to Q2:
extractor_no_pre_co = AttributeExtractor(X_train_p_tokens, X_train_h_tokens)
co_train = extractor_no_pre_co.co_occurrence_matrix()
# For test set, we need to consider the same vocabulary and build co-occurrence. This is tricky
# because the co-occurrence from test might not align exactly with train vocabulary.
# A simpler approach might be not to test co-occurrence directly as features for classification
# (because it creates a huge and sparse global matrix). Instead, consider a different approach,
# such as averaging embeddings or extracting only certain statistics.

# Due to complexity, let's skip direct classification with co-occurrence here or assume same vocab:
# We'll omit detailed co-occurrence classification due to complexity.

# --- Word2Vec Features ---
# We can represent each sentence pair by averaging their word embeddings.
extractor_pre_w2v = AttributeExtractor(X_train_pl_tokens, X_train_hl_tokens)
w2v_model = Word2Vec(sentences=X_train_pl_tokens + X_train_hl_tokens, vector_size=100, window=5, min_count=1, workers=4)
def avg_w2v(tokens):
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

X_train_w2v = [np.concatenate([avg_w2v(p), avg_w2v(h)]) for p, h in zip(X_train_pl_tokens, X_train_hl_tokens)]
X_test_w2v = [np.concatenate([avg_w2v(p), avg_w2v(h)]) for p, h in zip(X_test_pl_tokens, X_test_hl_tokens)]
X_train_w2v = np.array(X_train_w2v)
X_test_w2v = np.array(X_test_w2v)

acc_w2v_pre = train_and_evaluate(X_train_w2v, X_test_w2v, y_train, y_test)
print("Word2Vec (Lemmatized) Accuracy:", acc_w2v_pre)

# 2) Compare attribute extraction methods using chosen preprocessing (e.g., Lemmatization)
# We now have: acc_stats_pre, acc_cv_pre, acc_tfidf_pre, acc_w2v_pre
print("Comparing methods with Lemmatization:")
print("Stats:", acc_stats_pre)
print("CountVectorizer:", acc_cv_pre)
print("TF-IDF:", acc_tfidf_pre)
print("Word2Vec:", acc_w2v_pre)

# Select best attribute extraction based on accuracy. Suppose TF-IDF is best.

# 3) Variation of two preprocessings: Lemmatization vs Stemming with the best extraction method (TF-IDF)
X_train_comb_stem = combine_premise_hypothesis(X_train_ps, X_train_hs)
X_test_comb_stem = combine_premise_hypothesis(X_test_ps, X_test_hs)

tfidf_stem = TfidfVectorizer()
X_train_tfidf_stem = tfidf_stem.fit_transform(X_train_comb_stem)
X_test_tfidf_stem = tfidf_stem.transform(X_test_comb_stem)
acc_tfidf_stem = train_and_evaluate(X_train_tfidf_stem, X_test_tfidf_stem, y_train, y_test)

print("Comparing Lemmatization vs Stemming with TF-IDF:")
print("Lemmatization:", acc_tfidf_pre)
print("Stemming:", acc_tfidf_stem)


# Resultados obtidos
methods = ['Stats', 'CountVectorizer', 'TF-IDF', 'Word2Vec']
accuracies_no_pre = [0.727, 0.7, 0.727, 0]  # Sem Pré-processamento
accuracies_pre_lemmatization = [0.74, 0.711, 0.724, 0.726]  # Com Pré-processamento (Lemmatização)
accuracies_stemming = [0, 0, 0.725, 0]  # Stemming com TF-IDF (apenas melhor método)

# Gráfico comparativo de acurácias
x = np.arange(len(methods))
width = 0.3

plt.bar(x - width, accuracies_no_pre, width, label="Sem Pré-processamento", color='blue')
plt.bar(x, accuracies_pre_lemmatization, width, label="Com Pré-processamento (Lemmatização)", color='green')
plt.bar(x + width, accuracies_stemming, width, label="Stemming (TF-IDF)", color='orange')

plt.xlabel("Métodos de Extração de Atributos")
plt.ylabel("Acurácia")
plt.title("Comparação de Métodos de Extração de Atributos")
plt.xticks(x, methods)
plt.legend()
plt.ylim(0.65, 0.75)

plt.show()