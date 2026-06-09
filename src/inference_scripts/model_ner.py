# model.py
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# --- gleiche Architektur wie beim Training (mit Head 2) ---
class NERModel(nn.Module):
    def __init__(self, backbone_model_name, num_labels_1, num_labels_2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone_model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier_1 = nn.Linear(hidden_size, num_labels_1)
        self.classifier_2 = nn.Linear(hidden_size, num_labels_2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits1 = self.classifier_1(sequence_output)
        logits2 = self.classifier_2(sequence_output)
        return logits1, logits2

class Model:
    def __init__(self):
        # Kategorien für Head 1
        self.id2label = {0:"O", 1:"B-PER", 2:"I-PER", 3:"B-ORG", 4:"I-ORG", 5:"B-LOC", 6:"I-LOC"}
        self.num_labels_1 = len(self.id2label)
        self.num_labels_2 = 7  # Head 2 (nicht benutzt beim predict)
        self.model_name = "deepset/gbert-base"

        # Pfad zum gespeicherten Modell
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, "ner_model.pt")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Modell laden
        self.device = torch.device("cpu")
        self.model = NERModel(self.model_name, self.num_labels_1, self.num_labels_2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        tokens = list(x_test)

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits1, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits1, dim=-1)[0].cpu().numpy()

        word_ids = encoding.word_ids(batch_index=0)

        final_preds = []
        used_words = set()

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx in used_words:
                continue
            final_preds.append(self.id2label[preds[token_idx]])
            used_words.add(word_idx)

        # BIO-Fix (absolut notwendig)
        fixed = []
        prev = "O"
        for tag in final_preds:
            if tag.startswith("I-") and prev == "O":
                tag = "B-" + tag[2:]
            fixed.append(tag)
            prev = tag

        assert len(fixed) == len(tokens)
        return np.array(fixed)
