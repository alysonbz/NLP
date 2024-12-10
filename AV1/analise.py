import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def analyze_text_data(df: pd.DataFrame, text_column: str) -> None:
    print(f"Análise da coluna: {text_column}\n")


    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    df['char_count'] = df[text_column].apply(lambda x: len(str(x)))
    print(f"Número de textos: {len(df)}")
    print(f"Comprimento médio de palavras: {df['word_count'].mean():.2f}")
    print(f"Comprimento médio de caracteres: {df['char_count'].mean():.2f}")
    print(f"Texto mais longo (palavras): {df['word_count'].max()}")
    print(f"Texto mais longo (caracteres): {df['char_count'].max()}")

    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], kde=True, bins=30, color='blue')
    plt.title('Distribuição do Número de Palavras')
    plt.xlabel('Número de Palavras')
    plt.ylabel('Frequência')
    plt.show()

    text_data = " ".join(str(text) for text in df[text_column])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuvem de Palavras')
    plt.show()


def analyze_sentiment_data(df: pd.DataFrame, sentiment_column: str) -> None:

    print(f"Análise da coluna: {sentiment_column}\n")


    print(df[sentiment_column].describe())


    class_counts = df[sentiment_column].value_counts()
    print("\nDistribuição das Classes:\n", class_counts)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    plt.title('Distribuição das Classes de Sentimento')
    plt.xlabel('Classes de Sentimento')
    plt.ylabel('Contagem')
    plt.show()


    plt.figure(figsize=(8, 6))
    class_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(class_counts)))
    plt.title('Proporção de Classes')
    plt.ylabel('')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[sentiment_column], y=df['word_count'], palette='viridis')
    plt.title('Relação entre Sentimento e Número de Palavras')
    plt.xlabel('Pontuação de Sentimento')
    plt.ylabel('Número de Palavras')
    plt.show()


def analyze_class_relationship(df: pd.DataFrame, text_column: str, sentiment_column: str) -> None:

    print(f"Análise dos padrões textuais por classe em '{sentiment_column}'\n")

    unique_classes = df[sentiment_column].unique()
    for sentiment_class in unique_classes:
        class_text = " ".join(df[df[sentiment_column] == sentiment_class][text_column])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(class_text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Nuvem de Palavras - Classe {sentiment_class}')
        plt.show()


if __name__ == "__main__":

    dataset_path = r"C:\Users\MASTER\OneDrive\Área de Trabalho\NLP Aleky\NLP\AV1\brazilian_headlines_sentiments_preprocessed.csv"

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {dataset_path}")

    analyze_text_data(df, text_column='headlinePortuguese')
    analyze_sentiment_data(df, sentiment_column='sentimentScorePortuguese')
    analyze_class_relationship(df, text_column='headlinePortuguese', sentiment_column='sentimentScorePortuguese')


def analyze_sentiment_classes(df: pd.DataFrame, sentiment_column: str) -> None:

    # Agrupamento por faixas de sentimento
    df['sentiment_category'] = pd.cut(
        df[sentiment_column],
        bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
        labels=['Muito Negativo', 'Negativo', 'Neutro', 'Positivo', 'Muito Positivo']
    )

    print("\nDistribuição por Faixas de Sentimento:")
    print(df['sentiment_category'].value_counts())

    plt.figure(figsize=(8, 6))
    sns.countplot(x='sentiment_category', data=df,
                  order=['Muito Negativo', 'Negativo', 'Neutro', 'Positivo', 'Muito Positivo'], palette='coolwarm')
    plt.title('Distribuição por Categorias de Sentimento')
    plt.xlabel('Categoria de Sentimento')
    plt.ylabel('Contagem')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='sentiment_category', y='word_count', data=df, palette='coolwarm')
    plt.title('Comprimento do Texto por Categoria de Sentimento')
    plt.xlabel('Categoria de Sentimento')
    plt.ylabel('Número de Palavras')
    plt.show()

    rare_classes = df[sentiment_column].value_counts()[df[sentiment_column].value_counts() < 50].index
    print("\nClasses raras (menos de 50 ocorrências):")
    print(rare_classes)

    rare_texts = df[df[sentiment_column].isin(rare_classes)]
    print("\nExemplo de textos de classes raras:")
    print(rare_texts[['headlinePortuguese', sentiment_column]].head(10))


if __name__ == "__main__":
    analyze_sentiment_classes(df, sentiment_column='sentimentScorePortuguese')

from collections import Counter
from wordcloud import WordCloud


def analyze_words_by_category(df: pd.DataFrame, text_column: str, category_column: str):

    categories = df[category_column].unique()
    for category in categories:
        texts = " ".join(df[df[category_column] == category][text_column])
        tokens = texts.split()
        most_common = Counter(tokens).most_common(10)
        print(f"\nPalavras mais frequentes na categoria '{category}':")
        for word, freq in most_common:
            print(f"{word}: {freq}")

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Nuvem de Palavras - Categoria: {category}")
        plt.show()


if __name__ == "__main__":
    analyze_words_by_category(df, text_column='headlinePortuguese', category_column='sentiment_category')

