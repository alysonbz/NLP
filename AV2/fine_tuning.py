import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
from preprocessing import SentimentDataset

# Carregar dataset
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

# Carregar tokenizer e modelo pré-treinado
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Criar datasets usando a classe SentimentDataset
train_dataset = SentimentDataset(train_df['tweet_text'].tolist(), train_df['sentiment'].tolist(), tokenizer, max_len=128)
test_dataset = SentimentDataset(test_df["tweet_text"].tolist(), test_df["sentiment"].tolist(), tokenizer, max_len=128)

# Definir argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Ajustado para melhorar eficiência
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,  # Mantém apenas os 2 melhores checkpoints
)

# Criar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Ajustado para 2 épocas sem melhoria
)

# Executar treinamento
trainer.train()

# Avaliação do modelo
eval_results = trainer.evaluate()
print("Evaluation results: ", eval_results)

# Salvar modelo treinado
trainer.save_model("bert-base-uncased_model_fine_turning_train")
tokenizer.save_pretrained("bert-base-uncased_tokenizer_fine_turning_tokenizer")

