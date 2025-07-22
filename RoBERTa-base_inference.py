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
