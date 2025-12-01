import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from preprocessing import preprocess_text
from atribute_extraction import FeatureExtractor

os.makedirs("plots_attribute", exist_ok=True)

df = pd.read_csv("2019-05-28_portuguese_hate_speech_binary_classification.csv")
df = df.dropna(subset=["text"])

texts = [preprocess_text(t) for t in df["text"].astype(str)]

fx = FeatureExtractor(max_features=2000)

X_count = fx.fit_countvectorizer(texts)

terms = np.array(fx.count_vectorizer.get_feature_names_out())
freqs = np.asarray(X_count.sum(axis=0)).flatten()

idx = np.argsort(freqs)[-20:]
top_terms = terms[idx]
top_freqs = freqs[idx]

plt.figure(figsize=(12,6))
plt.bar(top_terms, top_freqs, color="dodgerblue")
plt.xticks(rotation=50)
plt.title("Top 20 Palavras — CountVectorizer")
plt.tight_layout()

plt.savefig("plots_attribute/top20_countvectorizer.png", dpi=300)
plt.show()

X_tfidf = fx.fit_tfidf(texts)

idf = fx.tfidf_vectorizer.idf_
terms = np.array(fx.tfidf_vectorizer.get_feature_names_out())

idx = np.argsort(idf)[:20]
top_terms = terms[idx]
top_idf = idf[idx]

plt.figure(figsize=(12,6))
plt.bar(top_terms, top_idf, color="orange")
plt.xticks(rotation=50)
plt.ylabel("IDF")
plt.title("Top 20 Palavras — TF-IDF (menor IDF = mais importante)")
plt.tight_layout()

plt.savefig("plots_attribute/top20_tfidf.png", dpi=300)
plt.show()

cooc, vocab = fx.build_cooccurrence_matrix(texts)

sums = cooc.sum(axis=1).sort_values(ascending=False)
top_words = sums.index[:15]
submatrix = cooc.loc[top_words, top_words]

plt.figure(figsize=(10,8))
sns.heatmap(submatrix, cmap="Blues")
plt.title("Heatmap — Matriz de Coocorrência (Top 15 Palavras)")
plt.tight_layout()

plt.savefig("plots_attribute/coocorrencia_heatmap.png", dpi=300)
plt.show()
fx.train_word2vec(texts)

words = fx.word2vec.wv.index_to_key[:200]
vecs = np.array([fx.word2vec.wv[w] for w in words])

from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(vecs)
sims = sim_matrix.flatten()

plt.figure(figsize=(10,6))
plt.hist(sims, bins=40, color="green")
plt.title("Distribuição de Similaridade (Word2Vec)")
plt.xlabel("Similaridade Coseno")
plt.ylabel("Frequência")
plt.tight_layout()

plt.savefig("plots_attribute/word2vec_similaridade.png", dpi=300)
plt.show()

print("Todos os gráficos foram salvos na pasta 'plots_attribute/'.")
