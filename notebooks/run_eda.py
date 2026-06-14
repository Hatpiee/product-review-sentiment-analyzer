"""
Standalone EDA script — equivalent to eda.ipynb but runs without Jupyter.
Run from project root: python notebooks/run_eda.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

os.makedirs('notebooks/plots', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('data/Reviews.csv')
print(f'Shape: {df.shape}')
print(f'\nDtypes:\n{df.dtypes}')
print(f'\nNull values:\n{df.isnull().sum()}')

# ── Rating distribution ───────────────────────────────────────────────────────
score_counts = df['Score'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=score_counts.index, y=score_counts.values, palette='Blues_d', ax=ax)
ax.set_title('Distribution of Star Ratings', fontsize=14)
ax.set_xlabel('Star Rating')
ax.set_ylabel('Count')
for i, v in enumerate(score_counts.values):
    ax.text(i, v + 500, f'{v:,}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('notebooks/plots/rating_distribution.png', dpi=150)
plt.close()
print('Saved: notebooks/plots/rating_distribution.png')

# ── Sentiment labels ──────────────────────────────────────────────────────────
def score_to_sentiment(score):
    if score in [1, 2]:
        return 'negative'
    elif score == 3:
        return 'neutral'
    return 'positive'

df['sentiment'] = df['Score'].apply(score_to_sentiment)
print(f'\nSentiment distribution:\n{df["sentiment"].value_counts()}')

# ── Sentiment distribution ────────────────────────────────────────────────────
order = ['positive', 'neutral', 'negative']
palette = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
counts = df['sentiment'].value_counts()
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(order, [counts[s] for s in order], color=[palette[s] for s in order],
       edgecolor='black', linewidth=0.5)
ax.set_title('Sentiment Label Distribution', fontsize=14)
ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
for i, v in enumerate([counts[s] for s in order]):
    ax.text(i, v + 500, f'{v:,}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('notebooks/plots/sentiment_distribution.png', dpi=150)
plt.close()
print('Saved: notebooks/plots/sentiment_distribution.png')

# ── Word clouds ───────────────────────────────────────────────────────────────
df = df.dropna(subset=['Text'])

pos_text = ' '.join(df[df['sentiment'] == 'positive']['Text'].astype(str).tolist())
neg_text = ' '.join(df[df['sentiment'] == 'negative']['Text'].astype(str).tolist())

for label, text, cmap in [('positive', pos_text, 'Greens'), ('negative', neg_text, 'Reds')]:
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap=cmap, max_words=200).generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud — {label.capitalize()} Reviews', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'notebooks/plots/wordcloud_{label}.png', dpi=150)
    plt.close()
    print(f'Saved: notebooks/plots/wordcloud_{label}.png')

# ── Review length distribution ────────────────────────────────────────────────
df['review_length'] = df['Text'].astype(str).apply(lambda x: len(x.split()))
fig, ax = plt.subplots(figsize=(10, 5))
for sentiment, color in [('positive', '#2ecc71'), ('neutral', '#95a5a6'), ('negative', '#e74c3c')]:
    subset = df[df['sentiment'] == sentiment]['review_length']
    ax.hist(subset[subset <= 300], bins=50, alpha=0.6, color=color, label=sentiment)
ax.set_title('Review Length Distribution by Sentiment', fontsize=14)
ax.set_xlabel('Word Count')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('notebooks/plots/review_length_distribution.png', dpi=150)
plt.close()
print('Saved: notebooks/plots/review_length_distribution.png')

print('\nEDA complete. All plots saved to notebooks/plots/')