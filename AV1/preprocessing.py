import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

# Carregar o dataset
df = pd.read_csv("C:/Users/laris/Downloads/dados nlp/pof_google_play_reviews.csv")

print("Colunas:")
print(df.columns)

# Distribuição do score
df.dropna(subset=['score'], inplace=True)
df = df[df['score'] != 0]
score_counts = df['score'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
bars = plt.bar(score_counts.index, score_counts.values, color='#E6E6FA', edgecolor='black')
plt.xlabel('Score')
plt.ylabel('Número de Avaliações')
plt.title('Distribuição das Avaliações (Score)')
plt.xticks(range(1, 6))

# Adicionar quantidade em cima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.show()

print("\nColuna com comentários:")
nulos = df['content'].isnull().sum()
print("\nNulos:", nulos)

df = df.dropna(subset=['content'])
print(f"Número de linhas após modificação: {len(df)}")

# Extração e balanceamento das classes (amostra de 10.000 linhas)
desired_count = 10000 // len(df['score'].value_counts())
rus = RandomUnderSampler(sampling_strategy={score: desired_count for score in df['score'].unique()}, random_state=1)
df, _ = rus.fit_resample(df, df['score'])

# Adicionar uma coluna 'id'
df['id'] = range(len(df))

df[['id','content', 'score']].to_csv('content_sem_trat.csv', index=False)  # salvar base sem tratamento

# Funções de pré-processamento
def remove_caract_transf_minusc(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove caracteres especiais e números
    text = re.sub(r'\s+', ' ', text).strip()  # Remove múltiplos espaços em branco
    text = text.lower()  # Converte para minúsculas
    return text

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["  
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002702-\U000027B0"  
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def remove_mencoes(text):
    return re.sub(r"@\w+", "", text)  # Remove menções

def remove_urls(text):
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

def tokenizar(text):
    return text.split()

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatizacao(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def preprocessamento_exemplo(text, use_lemmatization=True):
    print("Texto original:", text)

    text = remove_mencoes(text)
    print("Após remover menções:", text)

    text = remove_emojis(text)
    print("Após remover emojis:", text)

    text = remove_caract_transf_minusc(text)
    print("Após remover caracteres especiais e transformar em minúsculas:", text)

    text = remove_urls(text)
    print("Após remover URLs:", text)

    tokens = tokenizar(text)
    print("Após tokenizar:", tokens)

    tokens = remove_stopwords(tokens)
    print("Após remover stopwords:", tokens)

    if use_lemmatization:
        tokens = lemmatizacao(tokens)
        print("Após lematização:", tokens)
    else:
        tokens = stemming(tokens)
        print("Após stemming:", tokens)

    return tokens

def preprocessamento(text, use_lemmatization=True):
    text = remove_mencoes(text)
    text = remove_emojis(text)
    text = remove_caract_transf_minusc(text)
    text = remove_urls(text)
    tokens = tokenizar(text)
    tokens = remove_stopwords(tokens)

    if use_lemmatization:
        tokens = lemmatizacao(tokens)
    else:
        tokens = stemming(tokens)

    return tokens

# Exemplo de processamento passo a passo para uma linha específica
exemplo = df.iloc[293]['content']

print("\nExemplo de processamento passo a passo para a linha 293:")
preprocessamento_exemplo(exemplo, use_lemmatization=True)

# Aplicar o pré-processamento na coluna inteira
df.loc[:, 'content_lemmatized'] = df['content'].apply(lambda x: preprocessamento(x, use_lemmatization=True))

# Pré-processamento com stemming
df.loc[:, 'content_stemmed'] = df['content'].apply(lambda x: preprocessamento(x, use_lemmatization=False))

print("\nColunas: bruta / lematizada / stemizada:")
print(df[['content', 'content_lemmatized', 'content_stemmed']].head())

colunas_content = ['id','content', 'content_stemmed', 'content_lemmatized', 'score']
df[colunas_content].to_csv('content_lemmatized_stemmed.csv', index=False)
