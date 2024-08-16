import pandas as pd
from preprocessing import preprocess_text

def main():
    input_file = "C:\\Users\\laura\\Downloads\\buscape.csv"
    output_file = "C:\\Users\\laura\\Downloads\\buscape_processado.csv"

    try:
        # Carregar o dataset
        df = pd.read_csv(input_file)

        # Exibir as primeiras 5 linhas da coluna 'review_text' original
        print("Primeiras 5 linhas da coluna 'review_text' original:")
        print(df['review_text'].head())

        # Verificar as colunas disponíveis
        print("Colunas disponíveis:", df.columns)

        # Aplicar pré-processamento à coluna 'review_text'
        df['review_text_processed'] = df['review_text'].apply(
            lambda x: preprocess_text(x, use_stemming=False, use_lemmatization=True))

        # Exibir as primeiras 5 linhas do dataset processado
        print("Primeiras 5 linhas do dataset processado:")
        print(df[['review_text', 'review_text_processed']].head())

        df.to_csv(output_file, index=False)
        print(f"Arquivo processado salvo como '{output_file}'")

    except FileNotFoundError:
        print(f"O arquivo '{input_file}' não foi encontrado.")
    except Exception as e:
        print(f" erro: {e}")

if __name__ == "__main__":
    main()
