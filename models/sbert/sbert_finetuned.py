import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train_cleaned.csv')
df = df[df['EssaySet'] == 1].dropna(subset=['EssayText_Cleaned', 'FinalScore'])
df['score'] = df['FinalScore']

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

ref_texts = train_df[train_df['score'] == train_df['score'].max()]['EssayText_Cleaned'].tolist()

train_examples = []
for _, row in train_df.iterrows():
    for ref in ref_texts[:3]:
        label = float(row['score'] / train_df['score'].max())
        train_examples.append(InputExample(texts=[row['EssayText_Cleaned'], ref], label=label))

model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)

train_embeddings = model.encode(train_df['EssayText_Cleaned'].tolist(), show_progress_bar=True)
test_embeddings = model.encode(test_df['EssayText_Cleaned'].tolist(), show_progress_bar=True)

ref_mask = train_df['score'].values == train_df['score'].max()
ref_embeddings = train_embeddings[ref_mask]

similarities = cosine_similarity(test_embeddings, ref_embeddings).max(axis=1)

scaler = MinMaxScaler(feature_range=(0, 3))
predicted = scaler.fit_transform(similarities.reshape(-1, 1)).flatten()
actual = test_df['score'].values

pearson, _ = pearsonr(predicted, actual)
spearman, _ = spearmanr(predicted, actual)
rmse = np.sqrt(np.mean((predicted - actual) ** 2))

print(f"Pearson r: {pearson:.3f}")
print(f"Spearman p: {spearman:.3f}")
print(f"RMSE: {rmse:.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, alpha=0.5)
plt.plot([0, 3], [0, 3], 'r--', label='Idealna zhoda')
plt.xlabel('Ludske skore')
plt.ylabel('Predikované skore (SBERT fine-tuned)')
plt.title('SBERT fine-tuned vs. ľudské hodnotenie')
plt.legend()
plt.savefig('grafy/sbert_finetuned_vysledky.png')
plt.show()