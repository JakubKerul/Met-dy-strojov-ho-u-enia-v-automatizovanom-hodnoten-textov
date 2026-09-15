import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train_cleaned.csv')
df = df[df['EssaySet'] == 1].dropna(subset=['EssayText_Cleaned', 'FinalScore'])
df['score'] = df['FinalScore']

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_sentences = [text.lower().split() for text in train_df['EssayText_Cleaned']]
test_sentences = [text.lower().split() for text in test_df['EssayText_Cleaned']]

model = Word2Vec(sentences=train_sentences, vector_size=200, window=5, min_count=2, sg=1, seed=42)

tfidf = TfidfVectorizer()
tfidf.fit(train_df['EssayText_Cleaned'])
tfidf_vocab = tfidf.vocabulary_
idf_weights = tfidf.idf_

def get_weighted_vector(tokens, model, tfidf_vocab, idf_weights):
    vectors = []
    weights = []
    for token in tokens:
        if token in model.wv and token in tfidf_vocab:
            vectors.append(model.wv[token])
            weights.append(idf_weights[tfidf_vocab[token]])
    if not vectors:
        return np.zeros(model.vector_size)
    vectors = np.array(vectors)
    weights = np.array(weights)
    return np.average(vectors, axis=0, weights=weights)

train_vectors = np.array([get_weighted_vector(s, model, tfidf_vocab, idf_weights) for s in train_sentences])
test_vectors = np.array([get_weighted_vector(s, model, tfidf_vocab, idf_weights) for s in test_sentences])

ref_mask = train_df['score'].values == train_df['score'].max()
ref_vectors = train_vectors[ref_mask]

similarities = cosine_similarity(test_vectors, ref_vectors).max(axis=1)

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
plt.ylabel('Predikované skore (Word2Vec)')
plt.title('Word2Vec vs. ľudské hodnotenie')
plt.legend()
plt.savefig('grafy/word2vec_vysledky.png')
plt.show()