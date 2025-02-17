import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
from preprocessing import SentimentDataset


# Carregar dataset
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

# Vocab
vocab = {"<pad>": 0, "<unk>": 1}
train_dataset = SentimentDataset(train_df['review_text'].tolist(), train_df['rating'].tolist(), vocab)
test_dataset = SentimentDataset(test_df["review_text"].tolist(), test_df["rating"].tolist(), vocab)

# Configurar modelo
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Definir argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True
)

# Criar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]

)

# Executar treinamento
trainer.train()

# Avaliação do modelo
eval_results = trainer.evaluate()
print("Evaluation results: ", eval_results)

# Se a acurácia for calculada automaticamente:
if "eval_accuracy" in eval_results:
    print("Accuracy:", eval_results["eval_accuracy"])

# Salvar modelo treinado
trainer.save_model("bert-base-uncased_model_fine_turning_train")
tokenizer.save_pretrained("bert-base-uncased_tokenizer_fine_turning")