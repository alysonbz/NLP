# 6) Crie e print um dataframe de 4 linhas e 4 colunas e preencha com a execução das funções das questões 1,2,3 e 4.
# Cada coluna deve conter os resultados das funções. Utilize a função apply() no dataframe criado na questão 5.

import pandas as pd
from sent_1 import sentencas
from sent_2 import palavra
from sent_3 import numerico
from sent_4 import caixa_alta
from sent_5 import data

data['sentencas'] = data['texto'].apply(sentencas)
data['palavra'] = data['texto'].apply(palavra)
data['numerico'] = data['texto'].apply(numerico)
data['caixa_alta'] = data['texto'].apply(caixa_alta)

print(data.to_string())