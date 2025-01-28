import re

import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd


"""
## Anotações
O dataset em questão é o ASSIN (Avaliação de Similaridade Semântica e INferência textual), um corpus em português focado em duas tarefas principais: a identificação de relações de inferência textual (RTE, do inglês Recognizing Textual Entailment) e a medição de similaridade semântica entre pares de sentenças. Ele foi construído a partir de notícias em português europeu e português brasileiro, onde cada par de sentenças descreve um mesmo evento ou tópico, mas pode variar em conteúdo e forma linguística.

Significado de cada coluna:

sentence_pair_id: Um identificador numérico único para cada par de sentenças. Serve para referenciação e controle interno do dataset.

premise: Primeira sentença do par. Normalmente descreve um evento, fato ou situação.

hypothesis: Segunda sentença do par, também descrevendo um evento ou informação relacionada, podendo estar ou não semanticamente relacionada à premissa.

relatedness_score: Uma pontuação de similaridade semântica (entre 1 e 5) atribuída pelos anotadores humanos. Esse valor indica o quão semanticamente similares as duas sentenças são, indo de “não relacionadas” (1) até “muito similares” (5).

entailment_judgment: Rótulo que indica a relação de inferência entre as sentenças. Diferentemente do esquema clássico (entailment, contradiction, neutral), o ASSIN utiliza três categorias:

0 (none): Não existe relação de inferência entre as sentenças (elas não se contradizem nem se implicam).
1 (entailment): Uma das sentenças implica a outra, ou seja, o significado de uma está contido na outra.
2 (paraphrase): As duas sentenças transmitem praticamente a mesma informação, podendo ser consideradas paráfrases uma da outra.
"""
splits = {'train': 'full/train-00000-of-00001.parquet', 'test': 'full/test-00000-of-00001.parquet', 'validation': 'full/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/nilc-nlp/assin/" + splits["train"])

nlp = spacy.load("pt_core_news_sm")

# Lista de stopwords em português
stop_words = set(stopwords.words('portuguese'))

def remover_caracteres_especiais(texto: str) -> str:
    """Remove caracteres especiais, pontuação e múltiplos espaços."""
    # Remove caracteres não alfanuméricos, exceto espaços
    texto_limpo = re.sub(r'[^a-zA-Z0-9à-úÀ-Ú\s]', '', texto)
    # Remove espaços extras
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    return texto_limpo


def to_lower(texto: str) -> str:
    """Converte texto para letras minúsculas."""
    return texto.lower()


def remover_stopwords(tokens: list) -> list:
    """Remove stopwords da lista de tokens."""
    return [token for token in tokens if token not in stop_words]


def tokenizar(texto: str) -> list:
    """Transforma o texto em uma lista de tokens."""
    return word_tokenize(texto, language='portuguese')


def lematizar(tokens: list) -> list:
    """Lematiza uma lista de tokens usando o spaCy."""
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]


def remover_mencoes(texto: str, padrao: str = r'\b[A-ZÀ-Ú][a-zà-ú]+\b') -> str:
    """
    Exemplo de remoção de menções: remove palavras que começam com maiúscula
    (potencialmente nomes próprios).
    Ajustar regex conforme necessidade do dataset.
    """
    # Substitui por vazio todas as menções que correspondam ao padrão
    texto_sem_mencoes = re.sub(padrao, '', texto)
    # Remove espaços extras após substituição
    texto_sem_mencoes = re.sub(r'\s+', ' ', texto_sem_mencoes).strip()
    return texto_sem_mencoes


def preprocessar_texto(texto: str,
                       aplicar_remocao_mencoes=False,
                       aplicar_lematizacao=True) -> list:
    """
    Função principal de pré-processamento.
    - Converte para minúsculas
    - Remove caracteres especiais
    - (Opcional) Remove menções com base no padrão
    - Tokeniza
    - Remove stopwords
    - (Opcional) Lematiza tokens
    """

    # Converte para minúsculas
    texto = to_lower(texto)

    # Remove caracteres especiais
    texto = remover_caracteres_especiais(texto)

    # Remove menções se necessário
    if aplicar_remocao_mencoes:
        texto = remover_mencoes(texto)

    # Tokenização
    tokens = tokenizar(texto)

    # Remove stopwords
    tokens = remover_stopwords(tokens)

    # Lematização (opcional)
    if aplicar_lematizacao:
        tokens = lematizar(tokens)

    return tokens

# Visualizar exemplos de premise/hypothesis
print(f'Exemplo de linha da coluna premise \n {df['premise'].iloc[0]}')
print(f'Exemplo de linha da coluna hypothesis \n {df['hypothesis'].iloc[0]}')

# Aplica a função de pré-processamento na coluna 'premise'
df['premise_processed'] = df['premise'].apply(
    lambda x: preprocessar_texto(x, aplicar_remocao_mencoes=False, aplicar_lematizacao=True)
)

# Aplica a função de pré-processamento na coluna 'hypothesis'
df['hypothesis_processed'] = df['hypothesis'].apply(
    lambda x: preprocessar_texto(x, aplicar_remocao_mencoes=False, aplicar_lematizacao=True)
)


print(f'Exemplo de linha da coluna premise_processed \n {df['premise_processed'].iloc[1]}')
print(f'Exemplo de linha da coluna hypothesis_processed \n {df['hypothesis_processed'].iloc[1]}')

