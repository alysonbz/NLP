import stanza
import pandas as pd 

# Download do modelo em português
#stanza.download('pt')

nlp = stanza.Pipeline('pt')

texto = "Hoje eu fui ao mercado e comprei frutas frescas."
doc = nlp(texto)

# tokens, lemas e classes gramaticais
dados = []
for sent in doc.sentences:
    for word in sent.words:
        dados.append({
            "Texto": word.text,
            "Lema": word.lemma,
            "Classe gramatical (UPOS)": word.upos
        })

# tabela
tabela = pd.DataFrame(dados)
print(tabela)
