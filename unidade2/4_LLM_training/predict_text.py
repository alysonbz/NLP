import torch


def predict(text, model_path="text_classifier.pth"):
    model = TransformerEncoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = torch.argmax(outputs, dim=1).item()
    return prediction


text = "This is a test sentence."
print(f"Predicted class: {predict(text)}")