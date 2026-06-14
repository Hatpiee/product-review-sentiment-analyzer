"""
Compare rule-based sentiment tools (TextBlob, VADER) against:
  - the actual label derived from the star rating
  - the trained Logistic Regression model's prediction
"""

import re
import pandas as pd
import joblib
import nltk

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── NLTK downloads ─────────────────────────────────────────────────────────
for resource in ['vader_lexicon', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

SAMPLE_SIZE = 500
RANDOM_STATE = 42


# ── Helpers ────────────────────────────────────────────────────────────────
def score_to_sentiment(score):
    if score in [1, 2]:
        return 'negative'
    elif score == 3:
        return 'neutral'
    return 'positive'


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


def textblob_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    return 'neutral'


def vader_sentiment(text):
    scores = sia.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    return 'neutral'


# ── Load data & sample ─────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv('data/Reviews.csv').dropna(subset=['Text', 'Score'])
df['actual'] = df['Score'].apply(score_to_sentiment)
sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)

# ── Load trained model ─────────────────────────────────────────────────────
print('Loading trained model...')
clf = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

sample['clean_text'] = sample['Text'].apply(preprocess)
X = vectorizer.transform(sample['clean_text'])
sample['model_pred'] = clf.predict(X)

# ── Rule-based predictions ─────────────────────────────────────────────────
print('Running TextBlob and VADER...')
sample['textblob'] = sample['Text'].apply(textblob_sentiment)
sample['vader'] = sample['Text'].apply(vader_sentiment)

# ── Summary table ──────────────────────────────────────────────────────────
result = sample[['Score', 'actual', 'textblob', 'vader', 'model_pred']].copy()
result.columns = ['Star', 'Actual Label', 'TextBlob', 'VADER', 'LR Model']

pd.set_option('display.max_rows', 30)
print('\n' + '=' * 70)
print('SENTIMENT COMPARISON TABLE (sample of 30 rows)')
print('=' * 70)
print(result.head(30).to_string(index=False))

# ── Accuracy summary ───────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('ACCURACY vs ACTUAL LABEL')
print('=' * 70)
for method in ['TextBlob', 'VADER', 'LR Model']:
    acc = (result['Actual Label'] == result[method]).mean()
    print(f'  {method:<12}: {acc:.2%}')

# ── Agreement summary ─────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('ALL-THREE AGREEMENT RATE')
print('=' * 70)
all_agree = (
    (result['TextBlob'] == result['Actual Label']) &
    (result['VADER'] == result['Actual Label']) &
    (result['LR Model'] == result['Actual Label'])
)
print(f'  All three match actual: {all_agree.mean():.2%}')
