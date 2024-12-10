import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec

from preprocessing import Preprocessor
from atribute_extraction import AttributeExtractor

dataset_path = r"C:\Users\MASTER\OneDrive\Área de Trabalho\NLP Aleky\NLP\AV1\brazilian_headlines_sentiments_preprocessed.csv"
df = pd.read_csv(dataset_path)

preprocessor = Preprocessor()
extractor = AttributeExtractor(text_column='headlinePortuguese')

def evaluate_model(X, y, classifier, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return acc, report

df['preprocessed_text'] = preprocessor.preprocess_dataframe(df, 'headlinePortuguese')['headlinePortuguese']

df['sentiment_category'] = pd.cut(
    df['sentimentScorePortuguese'], bins=[-1, 0, 1], labels=['negative', 'positive']
)

def get_sentence_embedding(text, model):
    words = text.split()
    embeddings = [model.wv[word] for word in words if word in model.wv]
    if embeddings:
        return sum(embeddings) / len(embeddings)
    else:
        return [0] * model.vector_size

methods = ['count_vectorizer', 'tfidf', 'cooccurrence', 'word2vec']
results_b = {}

for method in methods:
    print(f"\nAnalisando método de extração: {method}")
    if method in ['count_vectorizer', 'tfidf']:
        X, _ = extractor.extract_count_vectorizer(df) if method == 'count_vectorizer' else extractor.extract_tfidf(df)
        if hasattr(X, "toarray"):
            X = pd.DataFrame(X.toarray())
    elif method == 'cooccurrence':
        X, _ = extractor.extract_cooccurrence_matrix(df)
        if hasattr(X, "toarray"):
            X = pd.DataFrame(X.toarray())
        if X.shape[0] != len(df):
            print(f"Erro: tamanhos inconsistentes entre X ({X.shape[0]}) e y ({len(df)}). Ignorando método cooccurrence.")
            continue
    elif method == 'word2vec':
        texts = [text.split() for text in df['preprocessed_text']]
        w2v_model = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=2, workers=4)
        X = pd.DataFrame([get_sentence_embedding(text, w2v_model) for text in df['preprocessed_text']])

    y = df['sentiment_category']
    if len(X) != len(y):
        print(f"Erro: tamanhos inconsistentes entre X ({len(X)}) e y ({len(y)}).")
        continue

    acc, report = evaluate_model(X, y, RandomForestClassifier())
    results_b[method] = (acc, report)

print("\nResultados do Item (b): Comparação entre métodos de extração\n")
for key, value in results_b.items():
    print(f"Método: {key}")
    print(f"Acurácia: {value[0]:.4f}")
    print(f"Relatório:\n{value[1]}")

preprocessing_variants = ['lemmatization', 'stemming']
results_c = {}

for variant in preprocessing_variants:
    print(f"\nAnalisando pré-processamento: {variant}")
    df['preprocessed_text'] = preprocessor.preprocess_dataframe(df, 'headlinePortuguese', method=variant)['headlinePortuguese']

    X, _ = extractor.extract_tfidf(df)
    if hasattr(X, "toarray"):
        X = pd.DataFrame(X.toarray())

    y = df['sentiment_category']
    if len(X) != len(y):
        print(f"Erro: tamanhos inconsistentes entre X ({len(X)}) e y ({len(y)}).")
        continue

    acc, report = evaluate_model(X, y, RandomForestClassifier())
    results_c[variant] = (acc, report)

print("\nResultados do Item (c): Comparação entre lematização e stemming\n")
for key, value in results_c.items():
    print(f"Pré-processamento: {key}")
    print(f"Acurácia: {value[0]:.4f}")
    print(f"Relatório:\n{value[1]}")
