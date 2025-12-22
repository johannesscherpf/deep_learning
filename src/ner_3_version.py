import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

model_name = "deepset/gbert-base"
train_file = "../data/train.tsv"
val_file = "../data/val.tsv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################### DATA #################################

def read_data(file_path):
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["idx", "token", "bio", "bio2"],
        dtype=str,
        keep_default_na=False,
        quoting=3,
        on_bad_lines='skip'
    )
    df = df[~df['token'].str.strip().str.startswith("#")]
    df = df[~df['token'].str.strip().str.startswith("http")]
    df = df[df['token'].str.strip() != ""]
    return df

def preprocess_data(df):
    sentences, labels_1, labels_2 = [], [], []
    cur_tokens, cur_labels1, cur_labels2 = [], [], []

    for idx, token, label1, label2 in zip(df["idx"], df["token"], df["bio"], df["bio2"]):
        if not idx.isdigit():
            if cur_tokens:
                sentences.append(cur_tokens)
                labels_1.append(cur_labels1)
                labels_2.append(cur_labels2)
                cur_tokens, cur_labels1, cur_labels2 = [], [], []
            continue
        cur_tokens.append(token)
        cur_labels1.append(label1)
        cur_labels2.append(label2)

    if cur_tokens:
        sentences.append(cur_tokens)
        labels_1.append(cur_labels1)
        labels_2.append(cur_labels2)

    return sentences, labels_1, labels_2

def clean_labels(labels_1, labels_2):
    valid_types = ["PER", "ORG", "LOC"]

    def clean_seq(seq):
        new_seq = []
        for label in seq:
            if label.startswith("B-") or label.startswith("I-"):
                for t in valid_types:
                    if label[2:].startswith(t):
                        new_seq.append(label[:2]+t)
                        break
                else:
                    new_seq.append("O")
            else:
                new_seq.append("O")
        return new_seq

    return [clean_seq(seq) for seq in labels_1], [clean_seq(seq) for seq in labels_2]

def turn_to_numeric(labels_1_clean, labels_2_clean):
    label_map = {"O":0,"B-PER":1,"I-PER":2,"B-ORG":3,"I-ORG":4,"B-LOC":5,"I-LOC":6}
    labels_1_int = [[label_map[l] for l in seq] for seq in labels_1_clean]
    labels_2_int = [[label_map[l] for l in seq] for seq in labels_2_clean]
    return labels_1_int, labels_2_int

############################### LAZY DATASET ###########################

class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.df = read_data(file_path)
        self.sentences, self.labels1, self.labels2 = preprocess_data(self.df)
        self.labels1_clean, self.labels2_clean = clean_labels(self.labels1, self.labels2)
        self.labels1_int, self.labels2_int = turn_to_numeric(self.labels1_clean, self.labels2_clean)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        labels1 = self.labels1_int[idx]
        labels2 = self.labels2_int[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels1, aligned_labels2 = [], []
        prev_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels1.append(-100)
                aligned_labels2.append(-100)
            elif word_idx != prev_idx:
                aligned_labels1.append(labels1[word_idx])
                aligned_labels2.append(labels2[word_idx])
            else:
                aligned_labels1.append(labels1[word_idx])
                aligned_labels2.append(labels2[word_idx])
            prev_idx = word_idx

        return (encoding['input_ids'].squeeze(),
        encoding['attention_mask'].squeeze(),
        torch.tensor(aligned_labels1),
        torch.tensor(aligned_labels2))

############################### TOKENIZER ##############################

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = NERDataset(train_file, tokenizer, max_len=128)
val_dataset = NERDataset(val_file, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

print("DataLoader fertig, Training kann starten")

################################# MODEL #####################################

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

num_labels_1 = 7
num_labels_2 = 7
model = NERModel(model_name, num_labels_1, num_labels_2).to(device)

################################# TRAIN #####################################
print("start training")


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

for epoch in range(3):
    model.train()
    total_loss = 0
    total_correct_1 = total_tokens_1 = 0
    total_correct_2 = total_tokens_2 = 0

    for step, batch in enumerate(train_loader):
        if step % 10 == 0:
            print(f"Epoch {epoch + 1} | Step {step}/{len(train_loader)}")

        optimizer.zero_grad()
        input_ids, attention_mask, labels1, labels2 = [b.to(device) for b in batch]
        logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)

        loss1 = criterion(logits1.view(-1, logits1.shape[-1]), labels1.view(-1))
        loss2 = criterion(logits2.view(-1, logits2.shape[-1]), labels2.view(-1))
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

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
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc1: {acc1:.4f} | Acc2: {acc2:.4f}")

################################# VALIDATION #################################

from seqeval.metrics import f1_score, classification_report

model.eval()

id2label = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC"
}

true_labels_1, pred_labels_1 = [], []
true_labels_2, pred_labels_2 = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels1, labels2 = [b.to(device) for b in batch]
        logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)

        preds1 = torch.argmax(logits1, dim=-1)
        preds2 = torch.argmax(logits2, dim=-1)

        for i in range(labels1.size(0)):
            true_seq1, pred_seq1 = [], []
            true_seq2, pred_seq2 = [], []

            for j in range(labels1.size(1)):
                if labels1[i, j].item() != -100:
                    true_seq1.append(id2label[labels1[i, j].item()])
                    pred_seq1.append(id2label[preds1[i, j].item()])

                if labels2[i, j].item() != -100:
                    true_seq2.append(id2label[labels2[i, j].item()])
                    pred_seq2.append(id2label[preds2[i, j].item()])

            true_labels_1.append(true_seq1)
            pred_labels_1.append(pred_seq1)
            true_labels_2.append(true_seq2)
            pred_labels_2.append(pred_seq2)

f1_1 = f1_score(true_labels_1, pred_labels_1)
f1_2 = f1_score(true_labels_2, pred_labels_2)

print(f"Validation F1 (Head 1): {f1_1:.4f}")
print(f"Validation F1 (Head 2): {f1_2:.4f}")

print("\nDetailed report (Head 1):")
print(classification_report(true_labels_1, pred_labels_1))

torch.save(model.state_dict(), "../models/ner_model.pt")
print("Model saved as ner_model.pt")
