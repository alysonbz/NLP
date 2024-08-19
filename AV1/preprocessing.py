import pandas as pd
import re
import string
from enelvo.normaliser import Normaliser
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Certifique-se de que os recursos do NLTK estão baixados
nltk.download('punkt')
nltk.download('stopwords')

# Caminhos para os datasets
caminho_treino = r'C:/Users/bianc/Downloads/Ciencia_de_dados/NLP/AV1/train.csv'
caminho_teste = r'C:/Users/bianc/Downloads/Ciencia_de_dados/NLP/AV1/test.csv'
caminho_validacao = r'C:/Users/bianc/Downloads/Ciencia_de_dados/NLP/AV1/validation.csv'

# Carregar datasets
redacao_treino = pd.read_csv(caminho_treino)
redacao_teste = pd.read_csv(caminho_teste)
redacao_validacao = pd.read_csv(caminho_validacao)


stop_words = {'e', 'de', 'a', 'os', 'as', 'ai', 'oh', 'ah'}
stop_palavras = set(stopwords.words('portuguese')).union(stop_words)

corrigir_palavras = {"deciquaver": "x", "vistoaaqêiele": "visto aquele","primadentrode": "prima dentro de","vidroquando": "vidro quando", "muitofiliz":"muito feliz",
                    "parasempri":"Para sempre","minhasamiga": "minhas amigas","animaistempo":"animais tempo","todobagusada":"toda bagunçada",
                    "ropabagusada":"roupa bagunçada","gardarroupas":"guarda-roupa","pvanatarate": "x","quidezaparesou":"que desapareceu","guidaoseipa":"x",
                    "ozanimaze": "os animais","paeuzaiaropa":"x","sidiseupou":"x","gritounauane": "gritou nauane","istounervoso": "estou nervoso",
                    "euistoumevoso":"eu estou nervoso","nusupesalhajk":"x","vamolutagoku": "vamos lutar goku", "agoraegostei": "agora eu gostei",
                    "supesalhaji":"x","poucodienegia": "pouco de energia","ugokusicasou":"x", "semaliquino":"ser maligno","seufimadeus": "x",
                    "quebracabeça": "quebra-cabeças", "fazercarinho": "fazer carinho","fazercarinho": "fazer carinho","mininotemoso": "menino teimoso",
                    "todosapagado":"todos apagados","leticiafalando":"leticia falando","nauasomprada": "não assopra","minhassprimas":"minhas primas",
                    "inpinotesadas":"hipnotizada","animaisquando":"animais quando","irmotambem": "irmos também","escondeesconde": "esconde-esconde",
                    "livrosinfantis":"livros infantis","ônibusprou":"x","paraoônibus": "para o ônibus","anosaviagem":"a nossa viajem",
                    "animaisalguns": "animais alguns","soltamelesubiu": "soltam ele subiu","jueljeiraetc":'x',"minhassprimas": "minhas primas",
                    "entãoechei":"então achei","nossoamigo":"nosso amigo","exspóricks":"x","minhaparede":"minha parede","aminhaparede": "a minha parede",
                    "muintocêto":"muito certo","maomematepo":"x","taredmágica":"tarde mágica","dosdessenhos":"do desenhos","quardaroupa":"guarda-roupa",
                    "uffahahah":"x","ninguenviu": "ninguém viu","asroupasdaga": "as roupas da","mariafalou":"maria falou", "vridos":"vidros", "deichou":"deixou",
                    "fiu":"fio","entaõ":"então","iso":"isso","prinquedo":"brinquedo", "en":"em", "pradentroagentecaiu": "para dentro a gente caiu", "si":"se"}


#normaliser = Normaliser()
def substituir_palavras_erradas(text):
    for palavra_errada, palavra_correta in corrigir_palavras.items():
        text = text.replace(palavra_errada, palavra_correta)
    return text

def remove_padrao_colchete_text(text):
    pattern = r'\[.*?\]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text.strip()

def remove_pontuacao(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def minusculas(text):
    return text.lower()

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_palavras]
    return ' '.join(filtered_words)

def remove_numeros(text):
    words = text.split()
    filtered_words = [word for word in words if not word.isdigit()]
    return ' '.join(filtered_words)


#def corrigir_palavras_enelvo(text):
    #return normaliser.normalise(text)


def preprocess_text(text):
    text = substituir_palavras_erradas(text)
    text = remove_padrao_colchete_text(text)
    text = remove_numeros(text)
    text = minusculas(text)
    text = remove_stopwords(text)
    text = remove_pontuacao(text)
    #text = corrigir_palavras_enelvo(text)

    return text

redacao_treino["aplicar"] = redacao_treino["essay"].apply(preprocess_text)
redacao_teste["aplicar"] = redacao_treino["essay"].apply(preprocess_text)
redacao_validacao["aplicar"] = redacao_validacao["essay"].apply(preprocess_text)

redacao_treino.to_csv(r'C:/Users/bianc/Downloads/Ciencia_de_dados/NLP/AV1/train_preprocessed.csv', index=False)
redacao_teste.to_csv(r'C:/Users/bianc/Downloads/Ciencia_de_dados/NLP/AV1/test_preprocessed.csv', index=False)
redacao_validacao.to_csv(r'C:/Users/bianc/Downloads/Ciencia_de_dados/NLP/AV1/validation_preprocessed.csv', index=False)
print(redacao_treino[['essay','aplicar']])
