# Product Review Sentiment Analyzer

A machine-learning pipeline that classifies Amazon product reviews as **Positive**, **Neutral**, or **Negative** using a Logistic Regression model trained on TF-IDF features, with a Streamlit web app for live inference.

---

## Tech Stack

| Layer | Library |
|---|---|
| Data manipulation | pandas, numpy |
| Text preprocessing | nltk (stopwords, WordNetLemmatizer) |
| Vectorization | scikit-learn TfidfVectorizer |
| Model | scikit-learn LogisticRegression |
| Rule-based comparison | TextBlob, VADER (nltk) |
| Visualization | matplotlib, seaborn, wordcloud |
| Model persistence | joblib |
| Web app | Streamlit |

---

## Folder Structure

```
product-review-sentiment-analyzer/
├── data/
│   └── Reviews.csv                      # Amazon Fine Food Reviews dataset
├── notebooks/
│   ├── eda.ipynb                        # EDA notebook (Jupyter)
│   ├── run_eda.py                       # EDA as a plain Python script
│   └── plots/
│       ├── rating_distribution.png
│       ├── sentiment_distribution.png
│       ├── wordcloud_positive.png
│       ├── wordcloud_negative.png
│       ├── review_length_distribution.png
│       └── confusion_matrix.png
├── src/
│   ├── train.py                         # Preprocessing + model training
│   ├── predict.py                       # Single-review inference script
│   └── compare_sentiment.py             # TextBlob vs VADER vs LR comparison
├── app/
│   └── app.py                           # Streamlit web application
├── models/
│   ├── sentiment_model.pkl              # Trained classifier (generated)
│   └── vectorizer.pkl                   # Fitted TF-IDF vectorizer (generated)
├── requirements.txt
└── README.md
```

---

## Dataset

**Amazon Fine Food Reviews** (~568,000 reviews)
Source: [https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

Place the downloaded `Reviews.csv` inside the `data/` folder before running any scripts.

**Sentiment label mapping:**

| Star Rating | Label    |
|-------------|----------|
| 4 – 5       | positive |
| 3           | neutral  |
| 1 – 2       | negative |

---

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Run Order

### 1. EDA (Exploratory Data Analysis)

**Option A — Jupyter notebook (opens in browser):**
```bash
.venv\Scripts\jupyter.exe notebook notebooks/eda.ipynb
```

**Option B — Plain Python script (no browser needed):**
```bash
python notebooks/run_eda.py
```

Both save all plots to `notebooks/plots/`.

---

### 2. Train the Model

```bash
python src/train.py
```

- Preprocesses the full dataset (lowercase, strip HTML, remove punctuation, stopwords, lemmatize)
- Fits a TF-IDF vectorizer (50K features, unigrams + bigrams)
- Trains Logistic Regression with stratified 80/20 split
- Prints accuracy and classification report
- Saves `notebooks/plots/confusion_matrix.png`
- Saves `models/sentiment_model.pkl` and `models/vectorizer.pkl`

---

### 3. Predict Sentiment (CLI)

```bash
python src/predict.py
```

Runs inference on hardcoded example reviews and prints the predicted sentiment + confidence scores.

---

### 4. Compare Rule-Based vs ML Sentiment

```bash
python src/compare_sentiment.py
```

Samples 500 reviews and compares TextBlob, VADER, and the trained LR model against the actual star-rating-derived labels.

---

### 5. Launch the Streamlit App

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** Run `src/train.py` first — the app requires the saved model files in `models/`.

---

## Streamlit App Features

- Text input box for typing or pasting a review
- One-click example reviews to test quickly
- Color-coded sentiment badge (green = positive, red = negative, gray = neutral)
- Probability bar chart for all three classes
- Word cloud reference images from training data
- Confusion matrix display

---

## Typical Results (full dataset)

| Metric | Value |
|---|---|
| Overall accuracy | ~92–94% |
| Positive F1 | ~0.96 |
| Neutral F1 | ~0.60 |
| Negative F1 | ~0.87 |

> The neutral class is harder to classify — fewer samples and more ambiguous language compared to strongly positive or negative reviews.