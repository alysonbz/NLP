from src.utils import load_movie_review_clean_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

corpus = load_movie_review_clean_dataset()

X = corpus['review']
y = corpus['sentiment']

X_train,  X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# step 01:
vectorizer = CountVectorizer()
x_train_counts = vectorizer.fit_transform(X_train)
vocab = vectorizer.get_feature_names_out()

# matriz de coocorrencia
cooc_matrix = (X_train.T @ x_train_counts).toarray()
cooc_df = pd.DataFrame(cooc_matrix, index=vocab, columns=vocab)

# step 02:
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

# step 03:
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# step 04
X_test_counts = vectorizer.transform(X_test)
X_test_tdfidf = tfidf_transformer.transform((X_test_counts))
y_pred = model.predict(X_test_tdfidf)

# Avaliação do modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de classificação:")
print(classification_report(y_test, y_pred))

cooc_df.to_csv("coocurrence_matrix.csv", index=True)