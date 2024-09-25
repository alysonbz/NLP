import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from imblearn.over_sampling import RandomOverSampler

# Limpando o arquivo CSV
def limpar_csv(caminho_entrada, caminho_saida):
    with open(caminho_entrada, 'r', newline='', encoding='utf-8') as infile, \
         open(caminho_saida, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            writer.writerow(row)

caminho_csv = r'C:\Users\laura\PycharmProjects\NLP\AV2\fv'
caminho_csv_limpo = r'C:\Users\laura\PycharmProjects\NLP\AV2\cleaned_fv.csv'
limpar_csv(caminho_csv, caminho_csv_limpo)

# Carregando os dados do CSV limpo
try:
    dados = pd.read_csv(caminho_csv_limpo, on_bad_lines='warn')
    dados = dados.dropna(subset=['review_text', 'polarity'])

    texts = dados['review_text'].tolist()
    labels = dados['polarity'].tolist()

    # classes únicas
    print(f'Classes presentes: {set(labels)}')

    # oversampling para balancear as classes
    if len(set(labels)) <= 1:
        print("Apenas uma classe presente. Pulando o oversampling.")
        texts_resampled, labels_resampled = texts, labels
    else:
        oversampler = RandomOverSampler(random_state=42)
        texts_resampled, labels_resampled = oversampler.fit_resample(pd.DataFrame(texts), labels)
        texts_resampled = texts_resampled[0].tolist()

    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(texts_resampled, labels_resampled, test_size=0.2, random_state=42)

    # modelo e o tokenizador
    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=2)

    # Tokenizar os textos
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512)

    # Criando dataset para o Trainer
    class TextDataset(torch.utils.data.Dataset):
        def _init_(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def _getitem_(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Garantir que os rótulos sejam do tipo long
            return item

        def _len_(self):
            return len(self.labels)

    train_dataset = TextDataset(train_encodings, y_train)
    test_dataset = TextDataset(test_encodings, y_test)

    # argumentos do treinamento
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # Treinando o modelo
    trainer.train()

    # Avaliando o modelo
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(-1)

    # Acurácia
    accuracy_llm = accuracy_score(y_test, predicted_labels)
    print(f"Acurácia do LLM: {accuracy_llm:.4f}")

except Exception as e:
    print(f'Ocorreu um erro: {e}')