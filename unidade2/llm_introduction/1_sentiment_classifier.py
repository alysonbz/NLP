# 1_sentiment_classifier.py

from transformers import pipeline

# Carrega o pipeline de classificação de sentimento usando um modelo pré-treinado BERT
classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

# Função para classificar o sentimento de uma lista de textos
def classify_sentiments(texts):
    results = classifier(texts)
    for text, result in zip(texts, results):
        label = result['label']
        score = result['score']
        print(f"Texto: {text}\nSentimento: {label}, Confiança: {score:.2f}\n")

if __name__ == "__main__":
    # Exemplo de textos para classificar
    sample_texts = [
        "Eu adorei esse filme!",
        "O serviço foi péssimo.",
        "A comida estava ok, nada de especial.",
        "Estou muito feliz com o atendimento.",
        "Não gostei da experiência."
    ]

    # Classifica os sentimentos dos textos
    classify_sentiments(sample_texts)