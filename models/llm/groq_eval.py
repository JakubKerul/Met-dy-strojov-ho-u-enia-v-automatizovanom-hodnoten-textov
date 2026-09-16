import pandas as pd
import numpy as np
from groq import Groq
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import time
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

df = pd.read_csv('data/train_cleaned.csv')
df = df[df['EssaySet'] == 1].dropna(subset=['EssayText_Cleaned', 'FinalScore'])
df['score'] = df['FinalScore']

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
test_sample = test_df.sample(n=50, random_state=42)

few_shot_examples = ""
for score in [0, 1, 2, 3]:
    examples = train_df[train_df['score'] == score].head(1)
    for _, row in examples.iterrows():
        few_shot_examples += f"Odpoveď: {row['EssayText_Cleaned']}\nSkóre: {int(row['score'])}\n\n"

def get_score(answer, few_shot_examples):
    prompt = f"""Si hodnotiteľ študentských odpovedí. Ohodnoť odpoveď na škále 0-3.
0 = nesprávna odpoveď
1 = čiastočne správna
2 = väčšinou správna
3 = úplne správna

Príklady:
{few_shot_examples}

Ohodnoť túto odpoveď iba číslom 0, 1, 2 alebo 3:
Odpoveď: {answer}
Skóre:"""

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        text = response.choices[0].message.content.strip()
        for char in text:
            if char in '0123':
                return float(char)
        return 1.5
    except Exception as e:
        print(f"ERROR: {e}")
        return 1.5

predicted = []
actual = test_sample['score'].values

for i, (_, row) in enumerate(test_sample.iterrows()):
    score = get_score(row['EssayText_Cleaned'], few_shot_examples)
    predicted.append(score)
    print(f"{i+1}/50 - Predikované: {score}, Skutočné: {row['score']}")
    time.sleep(0.5)

predicted = np.array(predicted)

pearson, _ = pearsonr(predicted, actual)
spearman, _ = spearmanr(predicted, actual)
rmse = np.sqrt(np.mean((predicted - actual) ** 2))

print(f"\nPearson r: {pearson:.3f}")
print(f"Spearman p: {spearman:.3f}")
print(f"RMSE: {rmse:.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, alpha=0.5)
plt.plot([0, 3], [0, 3], 'r--', label='Idealna zhoda')
plt.xlabel('Ludske skore')
plt.ylabel('Predikované skore (Qwen)')
plt.title('Qwen few-shot vs. ľudské hodnotenie')
plt.legend()
plt.savefig('grafy/qwen_vysledky.png')
plt.show()