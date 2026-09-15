import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train_cleaned.csv')
df = df[df['EssaySet'] == 1].dropna(subset=['EssayText_Cleaned', 'FinalScore'])
df['score'] = df['FinalScore']

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
svd = TruncatedSVD(n_components=100, random_state=42)

train_tfidf = vectorizer.fit_transform(train_df['EssayText_Cleaned'])
train_lsa = svd.fit_transform(train_tfidf)

test_tfidf = vectorizer.transform(test_df['EssayText_Cleaned'])
test_lsa = svd.transform(test_tfidf)

ref_mask = train_df['score'] == train_df['score'].max()
ref_vectors = train_lsa[ref_mask]

similarities = cosine_similarity(test_lsa, ref_vectors).max(axis=1)

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
plt.ylabel('Predikované skore (LSA)')
plt.title('LSA - sémantická podobnosť vs. ľudské hodnotenie')
plt.legend()
plt.savefig('grafy/lsa_vysledky.png')
plt.show()