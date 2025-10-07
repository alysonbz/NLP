import re
import nltk
import spacy
import pandas as pd
from nltk.stem import RSLPStemmer
from enelvo.normaliser import Normaliser

try:
    import language_tool_python
    LT_AVAILABLE = True
except Exception:
    LT_AVAILABLE = False

nltk.download("stopwords", quiet=True)
nltk.download("rslp", quiet=True)
nlp = spacy.load("pt_core_news_sm")

Hash_map = {
    r"\bvc\b": "você",
    r"\bce\b": "você",
    r"\bc\b": "você",
    r"\bamg\b": "amigo",
    r"\bamg.\b": "amigo.",
    r"\bpfv\b": "por favor",
    r"\bblz\b": "beleza",
    r"\btd\b": "tudo",
    r"\bq\b": "que",
    r"\bkd\b": "cadê",
    r"\bnaum\b": "não",
    r"\bn\b": "não",
    r"\bso\b": "só",
    r"\bsó\b": "só",
    r"\bti\b": "te",
    r"\bveju\b": "vejo",
    r"\bvejo\b": "vejo",
    r"\bcomu\b": "como",
    r"\bmais\b": "mas",   
    r"\blegau\b": "legal",
}

def apply_hash_map(text: str) -> str:
    for pat, repl in Hash_map.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text

normalizar = Normaliser(
    tokenizer="readable",
    capitalize_acs=True,
    capitalize_inis=True,
    capitalize_pns=True,
    sanitize=False
)

def grammar_fix(text: str) -> str:
    if not LT_AVAILABLE:
        return text
    try:
        tool = language_tool_python.LanguageTool('pt-BR')
        return tool.correct(text)
    except Exception:
        try:
            tool = language_tool_python.LanguageToolPublicAPI('pt-BR')
            return tool.correct(text)
        except Exception:
            return text

def lemma_pt_ajustada(token, doc):
    t = token.text.lower()
    if token.lemma_ == "ser":
        if any(tok.text.lower() == "para" for tok in doc):
            if t in {"fui", "foi", "vamos", "vou", "irei", "iremos", "iria"}:
                return "ir"
    return token.lemma_

texto = 'vc é um cara legau, mais eu so ti veju comu amg.'

print("=== Original ===")
print(texto, "\n")

t1 = apply_hash_map(texto)

t2 = normalizar.normalise(t1)

t3 = grammar_fix(t2)

print("=== Após Hash Map ===")
print(t1, "\n")
print("=== Após ENELVO ===")
print(t2, "\n")
print("=== Após LanguageTool ===")
print(t3, "\n")

doc = nlp(t3)
print("=== Tokens ===")
for tok in doc:
    print(tok.text)
print()

stopwords_pt = set(nltk.corpus.stopwords.words("portuguese"))
sem_stop = " ".join([w for w in t3.split() if w.lower() not in stopwords_pt])
print("=== Sem stopwords ===")
print(sem_stop, "\n")

stemmer = RSLPStemmer()
linhas = []
for tok in doc:
    linhas.append([tok.text, stemmer.stem(tok.text), lemma_pt_ajustada(tok, doc)])
df = pd.DataFrame(linhas, columns=["Token", "Stemming", "Lematização"])
print("=== Tabela ===")
print(df.to_string(index=False))