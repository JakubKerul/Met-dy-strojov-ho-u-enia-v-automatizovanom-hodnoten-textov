import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

df = pd.read_csv('data/train_cleaned.csv')
df = df[df['EssaySet'] == 1].dropna(subset=['EssayText_Cleaned', 'FinalScore'])
df['score'] = df['FinalScore']

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class AnswerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

train_dataset = AnswerDataset(train_df['EssayText_Cleaned'].tolist(), train_df['score'].tolist(), tokenizer)
val_dataset = AnswerDataset(val_df['EssayText_Cleaned'].tolist(), val_df['score'].tolist(), tokenizer)
test_dataset = AnswerDataset(test_df['EssayText_Cleaned'].tolist(), test_df['score'].tolist(), tokenizer)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

training_args = TrainingArguments(
    output_dir='models/bert/output',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

predictions = trainer.predict(test_dataset).predictions.flatten()
actual = test_df['score'].values

pearson, _ = pearsonr(predictions, actual)
spearman, _ = spearmanr(predictions, actual)
rmse = np.sqrt(np.mean((predictions - actual) ** 2))

print(f"Pearson r: {pearson:.3f}")
print(f"Spearman p: {spearman:.3f}")
print(f"RMSE: {rmse:.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(actual, predictions, alpha=0.5)
plt.plot([0, 3], [0, 3], 'r--', label='Idealna zhoda')
plt.xlabel('Ludske skore')
plt.ylabel('Predikované skore (BERT fine-tuned)')
plt.title('BERT fine-tuned vs. ľudské hodnotenie')
plt.legend()
plt.savefig('grafy/bert_finetuned_vysledky.png')
plt.show()