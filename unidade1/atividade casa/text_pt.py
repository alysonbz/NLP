import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
import spacy

# Baixe as dependências necessárias antes de executar o código:
# nltk.download('punkt')
# nltk.download('stopwords')

# Carregar modelos e bibliotecas necessárias
nlp = spacy.load("pt_core_news_sm")  # Modelo de SpaCy para português
stemmer = RSLPStemmer()  # Stemmer para português
stop_words = set(stopwords.words("portuguese"))  # Stop words em português


def corrigir_gramatica(texto):
    """
    Realiza uma correção básica de gramática no texto.
    Usa o modelo SpaCy para sugerir correções.
    """
    doc = nlp(texto)
    texto_corrigido = " ".join([token.text if not token.is_oov else token.lemma_ for token in doc])
    return texto_corrigido


def preprocessar_texto(texto):
    """
    Realiza o pré-processamento do texto:
    1. Tokeniza
    2. Remove stop words
    3. Aplica stemming e lematização
    """
    # Tokenização
    tokens = word_tokenize(texto, language="portuguese")

    # Remover stop words
    tokens_sem_stopwords = [token for token in tokens if token.lower() not in stop_words]

    # Aplicar stemming e lematização
    dados_processados = []
    for token in tokens_sem_stopwords:
        stem = stemmer.stem(token)  # Stemming
        lemma = nlp(token)[0].lemma_  # Lematização
        dados_processados.append({"token": token, "stemming": stem, "lematização": lemma})

    return dados_processados


def gerar_tabela(dados_processados):
    """
    Gera uma tabela com os resultados do processamento.
    """
    df = pd.DataFrame(dados_processados)
    return df


if __name__ == "__main__":
    # Texto de entrada com erros gramaticais
    texto_entrada = """
    O menino esta jogandu bola no park. Ela disia que ele não sabia joga direito mas a verdad era que ele estava melhorrando rapido.
    """

    # Passo a: Correção do texto
    texto_corrigido = corrigir_gramatica(texto_entrada)
    print("Texto Corrigido:\n", texto_corrigido)

    # Passos b, c, d: Tokenização, remoção de stop words, stemming e lematização
    dados_processados = preprocessar_texto(texto_corrigido)

    # Gerar tabela final
    tabela = gerar_tabela(dados_processados)

    # Mostrar tabela
    print("\nTabela de Tokens, Stemming e Lematização:")
    print(tabela)

    # Exportar tabela como CSV para análise adicional (opcional)
    tabela.to_csv("resultado_tokens.csv", index=False)
