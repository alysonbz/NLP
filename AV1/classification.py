# Packages -------------------------------------------------------------------------------------------------------------

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import spacy
import nltk

from NLP.AV1.vectorizer import vetorizar_tfidf, vetorizar_countvectorizer
from NLP.AV1.preprocessing import preprocessar_texto
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('rslp')
nltk.download('all')

stop_words = set(stopwords.words('portuguese'))
nlp = spacy.load('pt_core_news_sm')
stemmer = RSLPStemmer()


# Dados ----------------------------------------------------------------------------------------------------------------

df = pd.read_parquet("hf://datasets/AiresPucrs/sentiment-analysis-pt/data/train-00000-of-00001.parquet")
df_raw_sample = df.sample(n=1000, random_state=123).reset_index()


# Pré-processamento ----------------------------------------------------------------------------------------------------

df_raw_sample['text'] = df_raw_sample['text'].apply(preprocessar_texto)


# Vetorização ----------------------------------------------------------------------------------------------------------

docs_preproc = df_raw_sample['texto_preproc'].tolist()
matriz_tfidf_preproc, vocabulario_tfidf_preproc = vetorizar_tfidf(docs_preproc)

# docs_preproc = df_raw_sample['texto_preproc'].tolist()
# matriz_contagem_processado, vocabulario_contagem_processado = vetorizar_countvectorizer(docs_preproc)


# Classificação --------------------------------------------------------------------------------------------------------

mlp = MLPClassifier()

X_train, X_test, y_train, y_test = train_test_split(matriz_tfidf_preproc,
                                                    df_raw_sample['label'],
                                                    test_size=0.2,
                                                    random_state=42)


### Grid Search ########################################################################################################
param_grid = {
    'hidden_layer_sizes': [(100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Melhores parâmetros: ", grid_search.best_params_)
########################################################################################################################


clf = MLPClassifier(max_iter=200,
                    activation='relu',
                    hidden_layer_sizes=(50, 50),
                    learning_rate_init=0.001,
                    solver='lbfgs')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_preproc = accuracy_score(y_test, y_pred)

print("Acurácia com pré-processamento (TF-IDF):", accuracy_preproc)
print(classification_report(y_test, y_pred))



### Plot matriz
# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 17})
# plt.xlabel('Predição')
# plt.ylabel('Valor Real')
# plt.show()