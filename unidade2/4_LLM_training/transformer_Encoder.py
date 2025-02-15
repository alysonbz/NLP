from transformers import AutoModel
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        super(TransformerEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)
