import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

model_name = "deepset/gbert-base"
file_path = "../../data/train.tsv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_data(file_path):
    df = pd.read_csv(
        str(file_path),
        sep="\t",
        header=None,
        names=["idx", "token", "bio", "bio2"],  # adjust if needed
        engine="python",                         # safer than C engine
        quoting=3,                               # 3 = csv.QUOTE_NONE, ignore quotes
        dtype=str,
        keep_default_na=False,
        on_bad_lines='skip'                      # skip truly malformed lines
    )
    df = df[~df['token'].str.strip().str.startswith("#")]
    return df

def preprocess_data(df):
    sentences = []
    labels_1 = []
    labels_2 = []

    current_tokens = []
    current_labels_1 = []
    current_labels_2 = []

    for _, row in df.iterrows():
        # Header erkennen: irgendeine Spalte fängt mit "#" oder "[" an
        if str(row[0]).startswith("#") or str(row[0]).startswith("["):
            if current_tokens:
                sentences.append(current_tokens)
                labels_1.append(current_labels_1)
                labels_2.append(current_labels_2)
                current_tokens, current_labels_1, current_labels_2 = [], [], []
            continue

        token = row["token"]
        label_1 = row["bio"]
        label_2 = row["bio2"]

        if token == "":
            continue

        current_tokens.append(token)
        current_labels_1.append(label_1)
        current_labels_2.append(label_2)

    # flush last sentence
    if current_tokens:
        sentences.append(current_tokens)
        labels_1.append(current_labels_1)
        labels_2.append(current_labels_2)

    return sentences, labels_1, labels_2

def clean_labels(labels_1, labels_2):
    """
    Bereinigt Labels, sodass nur die Basisformen (B-/I-PER, B-/I-ORG, B-/I-LOC) übrig bleiben.
    """
    valid_types = ["PER", "ORG", "LOC"]

    def clean_seq(seq):
        new_seq = []
        for label in seq:
            if label.startswith("B-") or label.startswith("I-"):
                for t in valid_types:
                    if label[2:].startswith(t):
                        new_seq.append(label[:2] + t)
                        break
                else:
                    # falls es eine unbekannte Tag-Variante ist
                    new_seq.append("O")
            else:
                new_seq.append("O")  # z. B. "O"
        return new_seq

    labels_1_clean = [clean_seq(seq) for seq in labels_1]
    labels_2_clean = [clean_seq(seq) for seq in labels_2]

    return labels_1_clean, labels_2_clean

def tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def tokenize_and_align_labels(sentences, labels_1, labels_2, tokenizer):
    encodings = tokenizer(
        sentences,
        is_split_into_words=True,  # input is already tokenized
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    aligned_labels_1 = []
    aligned_labels_2 = []

    for i, sentence in enumerate(sentences):
        word_ids = encodings.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids_1 = []
        label_ids_2 = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids_1.append(-100)
                label_ids_2.append(-100)
            elif word_idx != previous_word_idx:
                label_ids_1.append(labels_1[i][word_idx])
                label_ids_2.append(labels_2[i][word_idx])
            else:
                label_ids_1.append(labels_1[i][word_idx])
                label_ids_2.append(labels_2[i][word_idx])
            previous_word_idx = word_idx

        aligned_labels_1.append(label_ids_1)
        aligned_labels_2.append(label_ids_2)

    return encodings, aligned_labels_1, aligned_labels_2

def turn_to_numeric(labels_1_clean, labels_2_clean):
    label_map_1 = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6}
    label_map_2 = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6}  # adjust if needed

    labels_1_int = [[label_map_1[l] for l in seq] for seq in labels_1_clean]
    labels_2_int = [[label_map_2[l] for l in seq] for seq in labels_2_clean]

    return labels_1_int, labels_2_int

############################### Run Preprocessing ####################################
df=read_data(file_path) # read
sentences, labels_1, labels_2 = preprocess_data(df) # preprocess
labels_1_clean, labels_2_clean = clean_labels(labels_1, labels_2) # clean labels
labels_1_int, labels_2_int = turn_to_numeric(labels_1_clean, labels_2_clean) # labels to ints
tokenizer_obj, _ = tokenizer(model_name)  # returns tokenizer and model
encodings, aligned_labels_1, aligned_labels_2 = tokenize_and_align_labels(sentences, labels_1_int, labels_2_int, tokenizer_obj)

# Convert input tensors
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# Convert aligned labels to tensors
labels1 = torch.tensor(aligned_labels_1)
labels2 = torch.tensor(aligned_labels_2)

dataset = TensorDataset(input_ids, attention_mask, labels1, labels2)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

#############################Modeling#########################################

class NERModel(nn.Module):
    def __init__(self, backbone_model_name, num_labels_1, num_labels_2):
        super(NERModel, self).__init__()
        # Load pretrained GBERT
        self.bert = AutoModel.from_pretrained(backbone_model_name)
        hidden_size = self.bert.config.hidden_size

        # Two linear heads for the two label sequences
        self.classifier_1 = nn.Linear(hidden_size, num_labels_1)
        self.classifier_2 = nn.Linear(hidden_size, num_labels_2)

    def forward(self, input_ids, attention_mask=None):
        # Get contextual embeddings from GBERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)

        # Predict for each label sequence
        logits_1 = self.classifier_1(sequence_output)  # shape: (batch_size, seq_len, num_labels_1)
        logits_2 = self.classifier_2(sequence_output)  # shape: (batch_size, seq_len, num_labels_2)

        return logits_1, logits_2

# Params
num_labels_1 = 7
num_labels_2 = 7

# Initialize Model
model = NERModel("deepset/gbert-base", num_labels_1, num_labels_2)
model.to(device)

#Train Model
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

model.train()
for epoch in range(3):
    total_loss = 0
    total_correct_1 = 0
    total_tokens_1 = 0
    total_correct_2 = 0
    total_tokens_2 = 0

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels1, labels2 = [b.to(device) for b in batch]

        logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss
        loss1 = criterion(logits1.view(-1, logits1.shape[-1]), labels1.view(-1))
        loss2 = criterion(logits2.view(-1, logits2.shape[-1]), labels2.view(-1))
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute token-level accuracy
        preds1 = torch.argmax(logits1, dim=-1)
        mask1 = labels1 != -100
        total_correct_1 += (preds1[mask1] == labels1[mask1]).sum().item()
        total_tokens_1 += mask1.sum().item()

        preds2 = torch.argmax(logits2, dim=-1)
        mask2 = labels2 != -100
        total_correct_2 += (preds2[mask2] == labels2[mask2]).sum().item()
        total_tokens_2 += mask2.sum().item()

    acc1 = total_correct_1 / total_tokens_1
    acc2 = total_correct_2 / total_tokens_2
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | "
          f"Acc1: {acc1:.4f} | Acc2: {acc2:.4f}")
