# TF-IDF.py
# Implementação MANUAL de TF-IDF para um corpus sintético de 5 linhas

from collections import Counter, defaultdict
import math
import re
import pandas as pd

#pré-processamento
def preprocess(texto: str) -> list[str]:
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)  # remove pontuação
    tokens = texto.split()
    stop = {
        "a","o","os","as","de","do","da","das","dos","e","é","em","um","uma",
        "no","na","nos","nas","que","com","para","por","se","ao","à","tem","já"
    }
    return [t for t in tokens if t not in stop]

#corpus
corpus = [
    "A análise de sentimentos ajuda a entender as opiniões dos consumidores.",
    "Modelos de linguagem natural podem detectar emoções em textos.",
    "Sentimentos negativos são frequentemente expressos em revisões de produtos.",
    "A análise de sentimentos pode melhorar a experiência do cliente.",
    "Avaliar o sentimento de uma frase pode revelar insights valiosos para empresas."
]

docs_tokens = [preprocess(doc) for doc in corpus]
N = len(docs_tokens)

vocab = sorted(set(t for toks in docs_tokens for t in toks))

#frequência do termo
tfs: list[Counter] = [Counter(toks) for toks in docs_tokens]

#em quantos documentos o termo aparece
df = defaultdict(int)
for t in vocab:
    df[t] = sum(1 for toks in docs_tokens if t in toks)

# escolha clássica: idf = ln( (N + 1) / (df + 1) ) + 1
idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in vocab}

#construir matriz TF-IDF
# TF simples (contagem) normalizada por comprimento do documento (tf / |doc|)
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
