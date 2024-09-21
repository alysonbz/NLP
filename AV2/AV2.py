from datasets import load_dataset
from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Carregar o dataset
ds = load_dataset("nilc-nlp/assin", "full")

# Selecionar uma amostra do dataset para exemplificação (para processamento rápido)
df = pd.DataFrame({
    'sentence1': ds['train']['sentence1'][:100],  # 100 primeiros textos
    'sentence2': ds['train']['sentence2'][:100],
    'entailment': ds['train']['entailment_judgment'][:100]  # Rótulos de classificação
})

# Ajustar os rótulos (0: Contradiction, 1: Entailment)
label_mapping = {'Entailment': 1, 'None': 0}
df['label'] = df['entailment'].map(label_mapping)

# Configurar o pipeline de classificação usando um modelo de LLM (BERT pré-treinado)
classifier = pipeline("text-classification", model="bert-base-multilingual-cased", return_all_scores=True)

# Função para aplicar o modelo ao dataset e obter previsões
def classify_with_llm(text1, text2):
    combined_text = f"Premise: {text1} Hypothesis: {text2}"
    result = classifier(combined_text)
    # Extrair o rótulo com maior probabilidade (0 ou 1)
    predicted_label = 1 if result[0][1]['score'] > result[0][0]['score'] else 0
    return predicted_label

# Aplicar o modelo às frases
df['predicted_label'] = df.apply(lambda x: classify_with_llm(x['sentence1'], x['sentence2']), axis=1)

# Calcular métricas de avaliação
accuracy = accuracy_score(df['label'], df['predicted_label'])
precision, recall, f1, _ = precision_recall_fscore_support(df['label'], df['predicted_label'], average='binary')

# Exibir os resultados
metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
})

print(metrics)
