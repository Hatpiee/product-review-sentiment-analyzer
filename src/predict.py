"""
Predict sentiment for a given review text using the trained model.
"""

import re
import joblib
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


def predict_sentiment(review_text):
    clf = joblib.load('models/sentiment_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')

    clean = preprocess(review_text)
    X = vectorizer.transform([clean])
    prediction = clf.predict(X)[0]
    probabilities = clf.predict_proba(X)[0]
    classes = clf.classes_
    confidence = probabilities[list(classes).index(prediction)]

    return prediction, confidence, dict(zip(classes, probabilities))


# ── Example reviews ────────────────────────────────────────────────────────
SAMPLE_REVIEWS = [
    "This product is absolutely amazing! Best purchase I've made all year. Highly recommend!",
    "Complete waste of money. Broke after two days and customer service was useless.",
    "It's okay. Nothing special, does what it says but doesn't exceed expectations.",
    "I love this! The flavor is incredible and the packaging arrived in perfect condition.",
    "Terrible quality. Smells bad and the color was completely different from the photo.",
]

if __name__ == '__main__':
    print('=' * 60)
    print('SENTIMENT PREDICTOR — example reviews')
    print('=' * 60)

    for i, review in enumerate(SAMPLE_REVIEWS, 1):
        sentiment, confidence, all_probs = predict_sentiment(review)

        color_map = {'positive': '\033[92m', 'negative': '\033[91m', 'neutral': '\033[90m'}
        reset = '\033[0m'
        color = color_map.get(sentiment, '')

        print(f'\nReview #{i}:')
        print(f'  Text      : {review[:80]}...' if len(review) > 80 else f'  Text      : {review}')
        print(f'  Sentiment : {color}{sentiment.upper()}{reset}')
        print(f'  Confidence: {confidence:.2%}')
        print(f'  All probs : { {k: f"{v:.2%}" for k, v in all_probs.items()} }')

    print('\n' + '=' * 60)