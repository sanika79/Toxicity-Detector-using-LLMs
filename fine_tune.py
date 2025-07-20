
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Load dataset
df = pd.read_csv("train.csv")
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df = df[['comment_text'] + label_cols].dropna()
print(df.shape)
df[label_cols] = df[label_cols].astype(int)

# Optional: Sample for faster testing
df = df.sample(5000, random_state=42).reset_index(drop=True)

# 2. Load model & tokenizer
model_name = "roberta-base"  
from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=6)

model.eval()

# 3. Predict function
def predict(texts, model, tokenizer):
    preds = []
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)  
            ## class 'transformers.modeling_outputs.SequenceClassifierOutput'> 
            ### output example - SequenceClassifierOutput(loss=None, logits=tensor([[-0.2325, -0.0248, -0.4093, -0.2885,  0.3855, -0.3670]]), hidden_states=None, attentions=None)
            ### len == 1
            logits = outputs.logits
            ## example of logits - tensor([[-0.5426, -0.2024, -0.2530, -0.4766,  0.3140, -0.3846]])
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            ## len of probs -- 6 -- which is = number of labels
            ## example of probs [0.5228014  0.3763359  0.5153118  0.44002044 0.5316354  0.5230016 ]
            preds.append(probs)   
            
    return np.array(preds)

# 4. Run predictions
probs = predict(df['comment_text'].tolist(), model, tokenizer)
threshold = 0.5
pred_labels = (probs > threshold).astype(int)

# 5. Evaluate
print("=== Multi-label Evaluation ===")
print("Micro F1-score:", f1_score(df[label_cols], pred_labels, average='micro'))
print("Macro F1-score:", f1_score(df[label_cols], pred_labels, average='macro'))
print("\nPer-label Classification Report:\n")
print(classification_report(df[label_cols], pred_labels, target_names=label_cols))


# 6. Fine-tuning on Training Data

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup

class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, label_cols, max_len=256):
        self.texts = df['comment_text'].tolist()
        self.labels = df[label_cols].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = labels
        return item

# Split into train/test
train_size = int(0.8 * len(df))
test_size = len(df) - train_size
train_df, test_df = random_split(df, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_dataset = ToxicDataset(train_df.dataset.iloc[train_df.indices], tokenizer, label_cols)
test_dataset = ToxicDataset(test_df.dataset.iloc[test_df.indices], tokenizer, label_cols)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=6)
model = model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 2
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Average training loss: {total_loss/len(train_loader):.4f}")

# 7. Evaluate fine-tuned model on test set

model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

all_probs = np.vstack(all_probs)
all_labels = np.vstack(all_labels)
pred_labels = (all_probs > 0.5).astype(int)

print("\n=== Fine-tuned Model Evaluation ===")
print("Micro F1-score:", f1_score(all_labels, pred_labels, average='micro'))
print("Macro F1-score:", f1_score(all_labels, pred_labels, average='macro'))
print("\nPer-label Classification Report:\n")
print(classification_report(all_labels, pred_labels, target_names=label_cols))
#
