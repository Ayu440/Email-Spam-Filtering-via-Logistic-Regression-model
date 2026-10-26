# Email Spam Filtering using Logistic Regression

A machine learning project that classifies emails as **Spam** or **Not Spam** using **TF-IDF text vectorization** and a **Logistic Regression** classifier. The model was trained in Google Colab and reaches about **98.8% accuracy** on a held-out test set.

The project is built around core, beginner-friendly data science concepts: dataset loading, imputation, outlier removal (Z-score), feature/target separation, and a stratified train-test split.

---

## Overview

Spam emails waste time and can carry phishing or scam content. This project trains a simple, fast, and interpretable text classifier that reads the raw text of an email and predicts whether it is spam, along with a confidence score.

## Repository Structure

```
├── app.py                  # Application that loads the saved model and predicts on emails
├── spam_model.pkl          # Trained Logistic Regression model
├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
└── README.md
```

> The training notebook (Google Colab) and `emails.csv` dataset can also be added to the repo for full reproducibility.

## Dataset

| Property | Value |
|---|---|
| File | `emails.csv` |
| Rows | 5,728 emails |
| Columns | `text` (email content), `spam` (label: `1` = spam, `0` = not spam) |
| Class split | 4,360 not spam (76.1%) / 1,368 spam (23.9%) |
| Missing values | None |

The dataset is moderately imbalanced, so a **stratified split** is used to keep the spam ratio the same in training and test sets.

## Workflow

1. **Dataset input**: load `emails.csv` with Pandas.
2. **Imputation**: missing text values are filled with an empty string; rows with a missing target label are dropped.
3. **Outlier removal (Z-score)**: email length (in characters) is computed and its Z-score calculated. Emails with `|Z| >= 3` (extremely long emails, **101 of them**) are removed, leaving **5,627** emails.
4. **Target and feature selection**: `X` = email `text`, `y` = `spam` label.
5. **Train-test split**: 80% train / 20% test (`random_state=42`, `stratify=y`), giving 4,501 training and 1,126 test emails.
6. **TF-IDF vectorization**: lowercase text, English stop words removed, top 10,000 features. The vectorizer is fit on training data only, to avoid data leakage.
7. **Model training**: `LogisticRegression(max_iter=1000, random_state=42)`.
8. **Evaluation**: accuracy, classification report, and confusion matrix.
9. **Model saving**: model and vectorizer exported with `joblib`.

## Results

**Accuracy: 98.85%**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Not Spam | 0.99 | 1.00 | 0.99 | 859 |
| Spam | 1.00 | 0.96 | 0.98 | 267 |

**Confusion matrix**

|  | Predicted Not Spam | Predicted Spam |
|---|---|---|
| **Actual Not Spam** | 858 | 1 |
| **Actual Spam** | 12 | 255 |

The model is very conservative: only 1 legitimate email was wrongly flagged as spam, while 12 spam emails slipped through.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (TF-IDF, Logistic Regression, metrics)
- SciPy (Z-score)
- Matplotlib, Seaborn (visualization)
- Joblib (model persistence)
- Google Colab

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ayu440/Email-Spam-Filtering-via-Logistic-Regression-model.git
cd Email-Spam-Filtering-via-Logistic-Regression-model
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
```

### 3. Run the app

```bash
python app.py
```

### Using the saved model directly

```python
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

email = "Congratulations! You have won a $10,000 prize. Click here to claim your reward!"
vec = vectorizer.transform([email])

prediction = model.predict(vec)[0]
confidence = model.predict_proba(vec)[0][prediction]

print("SPAM" if prediction == 1 else "NOT SPAM", f"({confidence:.2%})")
```

## Future Improvements

- Try other models (Naive Bayes, SVM, Random Forest) and compare.
- Use n-grams or word embeddings for richer features.
- Handle class imbalance (class weights or resampling) to improve spam recall.
- Add hyperparameter tuning with cross-validation.
- Deploy as a web app (Streamlit / Flask).

## Author

**Ayu440** — [GitHub](https://github.com/Ayu440)

## License

This project is open source and available for educational use.
