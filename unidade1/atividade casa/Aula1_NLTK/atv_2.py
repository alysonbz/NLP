
# 2. Mostre uma tabela com resultados da pesquisa sobre os significados das siglas POS tagging que foram calculadas pela spaCy

import spacy
import pandas as pd

nlp = spacy.load("pt_core_news_sm")

texto = "O gato preto correu rapidamente para o jardim do Brasil."

doc = nlp(texto)

rows = []
for token in doc:
    rows.append({
        "Token": token.text,
        "POS": token.pos_,
        "Significado": spacy.explain(token.pos_)
    })

df = pd.DataFrame(rows)

print(df)
