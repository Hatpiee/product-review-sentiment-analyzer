import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── NLTK downloads ────────────────────────────────────────────────────────────
for resource in ['stopwords', 'wordnet', 'omw-1.4', 'punkt']:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

os.makedirs('models', exist_ok=True)
os.makedirs('notebooks/plots', exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def score_to_sentiment(score):
    if score in [1, 2]:
        return 'negative'
    elif score == 3:
        return 'neutral'
    return 'positive'


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)          # strip HTML
    text = re.sub(r'[^a-z\s]', ' ', text)         # remove punctuation / digits
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# ── Load & label ─────────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv('data/Reviews.csv')
print(f'Dataset shape: {df.shape}')

df = df.dropna(subset=['Text', 'Score'])
df['sentiment'] = df['Score'].apply(score_to_sentiment)

print('Sentiment distribution:')
print(df['sentiment'].value_counts())

# ── Preprocess ────────────────────────────────────────────────────────────────
print('\nPreprocessing text (this may take a few minutes on the full dataset)...')
df['clean_text'] = df['Text'].apply(preprocess)

# ── TF-IDF ───────────────────────────────────────────────────────────────────
print('Vectorizing...')
vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# ── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}')

# ── Train ─────────────────────────────────────────────────────────────────────
print('\nTraining Logistic Regression...')
clf = LogisticRegression(
    max_iter=1000,
    solver='saga',
    C=1.0,
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {acc:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))

# ── Confusion matrix plot ─────────────────────────────────────────────────────
labels = ['negative', 'neutral', 'positive']
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title(f'Confusion Matrix  (Accuracy: {acc:.4f})', fontsize=13)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('notebooks/plots/confusion_matrix.png', dpi=150)
plt.close()
print('Saved: notebooks/plots/confusion_matrix.png')

# ── Persist artifacts ─────────────────────────────────────────────────────────
joblib.dump(clf, 'models/sentiment_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print('\nSaved: models/sentiment_model.pkl')
print('Saved: models/vectorizer.pkl')
print('\nDone!')
