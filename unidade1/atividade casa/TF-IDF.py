from collections import Counter, defaultdict
import math
import re
import pandas as pd

def preprocess(texto: str) -> list[str]:
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)  
    tokens = texto.split()
    stop = {
        "a","o","os","as","de","do","da","das","dos","e","é","em","um","uma",
        "no","na","nos","nas","que","com","para","por","se","ao","à","tem","já"
    }
    return [t for t in tokens if t not in stop]

corpus = [
    "Eu gosto de PLN; PLN é incrível para análise de textos.",
    "Modelos de linguagem aprendem padrões em grandes corpora.",
    "TF IDF destaca termos importantes em documentos.",
    "CountVectorizer conta palavras; TF IDF pondera pela frequência inversa.",
    "Pré-processamento como remoção de stopwords ajuda os modelos."
]

docs_tokens = [preprocess(doc) for doc in corpus]
N = len(docs_tokens)

vocab = sorted(set(t for toks in docs_tokens for t in toks))

tfs: list[Counter] = [Counter(toks) for toks in docs_tokens]

df = defaultdict(int)
for t in vocab:
    df[t] = sum(1 for toks in docs_tokens if t in toks)

idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in vocab}

rows = []
for d, tf_counter in enumerate(tfs):
    tam_doc = sum(tf_counter.values()) or 1
    linha = []
    for t in vocab:
        tf_norm = tf_counter[t] / tam_doc
        linha.append(tf_norm * idf[t])
    rows.append(linha)

df_tfidf = pd.DataFrame(rows, columns=vocab, index=[f"doc_{i+1}" for i in range(N)])
pd.set_option("display.max_columns", None)
print("\n=== Matriz TF-IDF (implementação manual) ===")
print(df_tfidf.round(4))