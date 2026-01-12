from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import pandas as pd
from processing import ler_datasets
from dotenv import load_dotenv

load_dotenv()

class CohesionOutput(BaseModel):
    cohesion: Literal[1, 2, 3, 4, 5] = Field(
        description="Nota de 1 a 5 indicando o nível de coesão textual"
    )
    justificativa: str = Field(
        description="Justificativa breve e objetiva da classificação"
    )

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Você é um especialista em análise de discurso com ênfase em coesão textual.
Sua tarefa é avaliar a coesão de um ensaio transcrito originalmente de forma manuscrita.

Coesão textual se refere à forma como as ideias são conectadas por meio de:
- conectores
- progressão lógica
- referências internas
- consistência semântica
- ausência de saltos arbitrários

A nota deve seguir a escala ordinal:
1 = muito baixa coesão
2 = baixa coesão
3 = média coesão
4 = alta coesão
5 = muito alta coesão

O ensaio vem de um dataset real que contém tokens de marcação da transcrição manuscrita. Esses tokens NÃO devem influenciar negativamente na avaliação da coesão. Eles apenas sinalizam aspectos físicos do manuscrito. Seus significados são:

[P], [ p ], {{p}}, etc → novo parágrafo  
[S], {{s}} → símbolo  
[T], {{t}} → início do título  
[R], [X], {{x}}, etc → rasura do manuscrito  
[?], {{ ? }} → token desconhecido  
[LC], [LT], etc → linha não reta na escrita

Ignore todos os tokens acima no julgamento da coesão, mas mantenha o texto remanescente para análise semântica.

Sua resposta deve seguir estritamente o schema de saída fornecido.
Não gere qualquer texto fora do JSON final.
        """.strip()
    ),
    (
        "user",
        """
Avalie a coesão do seguinte ensaio e retorne apenas o JSON:

{texto}
        """.strip()
    )
])

structured_llm = llm.with_structured_output(CohesionOutput)
chain = prompt | structured_llm

train, test, validation = ler_datasets()

TEXT_COLUMN = "essay"

results = []

for _, row in tqdm(test.iterrows(), total=len(test)):
    texto = row[TEXT_COLUMN]

    try:
        output = chain.invoke({"texto": texto})
        results.append({
            "cohesion_pred": output.cohesion,
            "cohesion_justification": output.justificativa
        })
    except Exception as e:
        results.append({
            "cohesion_pred": None,
            "cohesion_justification": f"ERROR: {e}"
        })

df_results = pd.concat(
    [test.reset_index(drop=True), pd.DataFrame(results)],
    axis=1
)

df_results = df_results[['cohesion', 'cohesion_pred', 'cohesion_justification']]
df_results.to_csv("results.csv", index=False)
