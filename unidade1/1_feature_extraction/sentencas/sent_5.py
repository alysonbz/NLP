# 5) Crie um dataframe de 1 coluna e 4 linhas. Este contém os textos testados nas questões anteriores.
import pandas as pd
textos = [
    "o rato roeu a roupa do rei de roma",
    "O que é que Cacá quer? Cacá quer caqui. Qual caqui que Cacá quer? Cacá quer qualquer caqui",
    "1 elefante incomoda muita gente, 2 elefantes incomodam muito mais, 3 elefantes incomodam muita gente, 4 elefantes incomodam muito mais",
    "O QUE É QUE CACÁ QUER? CACÁ QUER CAQUI. QUAL CAQUI QUE CACÁ QUER? CACÁ QUER QUALQUER CAQUI"
]

data = pd.DataFrame(textos, columns=['texto'])
data.to_csv('textos_dataset.csv')