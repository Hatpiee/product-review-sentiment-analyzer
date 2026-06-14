import re
import os
import sys
import joblib
import nltk
import numpy as np
import streamlit as st
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── NLTK downloads ────────────────────────────────────────────────────────────
for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ── Paths (works whether run from repo root or /app) ─────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sentiment_model.pkl')
VEC_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
PLOTS_DIR = os.path.join(BASE_DIR, 'notebooks', 'plots')


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


@st.cache_resource(show_spinner='Loading model...')
def load_model():
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return clf, vectorizer


def predict(text, clf, vectorizer):
    clean = preprocess(text)
    X = vectorizer.transform([clean])
    label = clf.predict(X)[0]
    probs = clf.predict_proba(X)[0]
    classes = list(clf.classes_)
    confidence = probs[classes.index(label)]
    return label, confidence, dict(zip(classes, probs))


def sentiment_badge(label):
    styles = {
        'positive': ('background:#d4edda;color:#155724;border:1px solid #c3e6cb;',
                     'POSITIVE'),
        'negative': ('background:#f8d7da;color:#721c24;border:1px solid #f5c6cb;',
                     'NEGATIVE'),
        'neutral':  ('background:#e2e3e5;color:#383d41;border:1px solid #d6d8db;',
                     'NEUTRAL'),
    }
    style, text = styles.get(label, ('', label.upper()))
    return (
        f'<div style="display:inline-block;padding:10px 22px;border-radius:8px;'
        f'font-size:1.4rem;font-weight:700;{style}">{text}</div>'
    )


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Product Review Sentiment Analyzer',
    page_icon='🛍️',
    layout='wide',
)

st.markdown(
    """
    <style>
    .main-title {font-size:2.2rem;font-weight:800;color:#2c3e50;margin-bottom:0}
    .sub-title  {font-size:1rem;color:#7f8c8d;margin-top:4px;margin-bottom:1.5rem}
    .section-header {font-size:1.1rem;font-weight:600;color:#34495e;margin-top:1rem}
    .stTextArea textarea {font-size:0.95rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🛍️ Product Review Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Trained on Amazon Fine Food Reviews — '
    'Logistic Regression + TF-IDF</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('About')
    st.markdown(
        """
        **Model:** Logistic Regression (multinomial)
        **Features:** TF-IDF (unigrams + bigrams, 50K vocab)
        **Labels:** Positive · Neutral · Negative
        **Dataset:** Amazon Fine Food Reviews (~568K reviews)

        **Label mapping**
        | Stars | Sentiment |
        |-------|-----------|
        | 4–5   | Positive  |
        | 3     | Neutral   |
        | 1–2   | Negative  |
        """
    )
    st.divider()
    st.caption('Built with Streamlit · scikit-learn · NLTK')

# ── Model loading ─────────────────────────────────────────────────────────────
models_missing = not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH))
if models_missing:
    st.error(
        '**Model files not found.**  '
        'Run `python src/train.py` from the project root first, then restart this app.'
    )
    st.stop()

clf, vectorizer = load_model()

# ── Main two-column layout ────────────────────────────────────────────────────
col_input, col_result = st.columns([1.2, 1], gap='large')

with col_input:
    st.markdown('<p class="section-header">Enter a product review:</p>', unsafe_allow_html=True)
    review_text = st.text_area(
        label='Review text',
        label_visibility='collapsed',
        placeholder='e.g. "This coffee is absolutely amazing — best I\'ve ever tasted!"',
        height=200,
    )

    example_reviews = {
        '⭐⭐⭐⭐⭐ Great product': 'Absolutely love this product! Works perfectly and arrived quickly. Will buy again!',
        '⭐ Terrible quality':    'Complete waste of money. Broke after one use and smells awful. Avoid at all costs.',
        '⭐⭐⭐ So-so':           'It is okay, nothing exceptional. Does the job but I expected more for the price.',
    }
    st.markdown('<p class="section-header">Or try an example:</p>', unsafe_allow_html=True)
    chosen = st.selectbox('Example reviews', ['— select —'] + list(example_reviews.keys()),
                          label_visibility='collapsed')
    if chosen != '— select —':
        review_text = example_reviews[chosen]
        st.info(f'**Loaded:** {review_text}')

    analyze_btn = st.button('Analyze Sentiment', type='primary', use_container_width=True)

with col_result:
    st.markdown('<p class="section-header">Result:</p>', unsafe_allow_html=True)

    if analyze_btn or (review_text and chosen != '— select —'):
        text_to_analyze = review_text.strip()
        if not text_to_analyze:
            st.warning('Please enter a review first.')
        else:
            with st.spinner('Analyzing...'):
                label, confidence, all_probs = predict(text_to_analyze, clf, vectorizer)

            st.markdown(sentiment_badge(label), unsafe_allow_html=True)
            st.markdown(f'**Confidence:** `{confidence:.2%}`')
            st.divider()
            st.markdown('**Probability breakdown:**')

            for sent in ['positive', 'neutral', 'negative']:
                prob = all_probs.get(sent, 0)
                color = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}[sent]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
                    f'<span style="width:70px;font-size:0.85rem">{sent.capitalize()}</span>'
                    f'<div style="flex:1;background:#ecf0f1;border-radius:4px;height:18px">'
                    f'<div style="width:{prob*100:.1f}%;background:{color};height:100%;border-radius:4px"></div>'
                    f'</div>'
                    f'<span style="font-size:0.85rem;width:45px;text-align:right">{prob:.1%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info('Enter a review and click **Analyze Sentiment**.')

# ── Word cloud reference section ──────────────────────────────────────────────
st.divider()
st.subheader('Reference Word Clouds from Training Data')

wc_pos_path = os.path.join(PLOTS_DIR, 'wordcloud_positive.png')
wc_neg_path = os.path.join(PLOTS_DIR, 'wordcloud_negative.png')

if os.path.exists(wc_pos_path) and os.path.exists(wc_neg_path):
    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.image(Image.open(wc_pos_path), caption='Positive Reviews', use_container_width=True)
    with wc_col2:
        st.image(Image.open(wc_neg_path), caption='Negative Reviews', use_container_width=True)
else:
    st.caption(
        'Word cloud images not found. Run `notebooks/eda.ipynb` '
        'to generate them in `notebooks/plots/`.'
    )

# ── Confusion matrix reference ────────────────────────────────────────────────
cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
if os.path.exists(cm_path):
    st.divider()
    st.subheader('Model Confusion Matrix (Test Set)')
    _, cm_col, _ = st.columns([1, 2, 1])
    with cm_col:
        st.image(Image.open(cm_path), use_container_width=True)