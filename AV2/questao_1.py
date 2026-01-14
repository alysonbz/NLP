import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import preprocess_text

OLLAMA_URL = "http://localhost:11434/api/generate"
SAMPLE_SIZE = 30 

df = pd.read_csv("2019-05-28_portuguese_hate_speech_binary_classification.csv")
df_hate = df[df['hatespeech_comb'] == 1].sample(SAMPLE_SIZE, random_state=42)
df_no_hate = df[df['hatespeech_comb'] == 0].sample(SAMPLE_SIZE, random_state=42)
test_sample = pd.concat([df_hate, df_no_hate]).sample(frac=1, random_state=42)

print("Processando baseline AV1...")
X_train_raw, _, y_train, _ = train_test_split(df['text'], df['hatespeech_comb'], test_size=0.2, random_state=42)

X_train_processed = [preprocess_text(t, use_stemming=True, use_lemmatization=False) for t in X_train_raw]

vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train_processed)
clf = LogisticRegression()
clf.fit(X_train_tfidf, y_train)

X_test_processed = [preprocess_text(t, use_stemming=True, use_lemmatization=False) for t in test_sample['text']]
test_sample['pred_av1'] = clf.predict(vectorizer.transform(X_test_processed))

def classify_local(text, model_name):
    prompt = f"Analise se o tweet abaixo é discurso de ódio. Responda APENAS '1' para sim ou '0' para não.\n\nTweet: {text}"
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        res_text = response.json()['response'].strip()
        return 1 if '1' in res_text[:3] else 0
    except Exception as e:
        print(f"Erro no modelo {model_name}: {e}")
        return 0

print(f"Iniciando classificação local em {len(test_sample)} exemplos...")

print("-> Rodando Llama3...")
test_sample['pred_llama3'] = test_sample['text'].apply(lambda x: classify_local(x, "llama3"))

print("-> Rodando Gemma2...")
test_sample['pred_gemma2'] = test_sample['text'].apply(lambda x: classify_local(x, "gemma2"))

for col in ['pred_av1', 'pred_llama3', 'pred_gemma2']:
    acc = accuracy_score(test_sample['hatespeech_comb'], test_sample[col])
    f1 = f1_score(test_sample['hatespeech_comb'], test_sample[col])
    print(f"\nModel: {col.upper()}\nAcc: {acc:.4f} | F1: {f1:.4f}")

test_sample.to_csv("resultado_comparativo_local_2.csv", index=False)
print("\nArquivo 'resultado_comparativo_local.csv' gerado para análise qualitativa.")