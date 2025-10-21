# Sentiment Analysis using NLTK and scikit-learn

This Python program performs sentiment analysis on the NLTK `movie_reviews` dataset using two types of vectorizers (CountVectorizer and TF-IDF) and two classifiers (Naive Bayes and LinearSVC). The goal is to compare and evaluate performance across these combinations.

---

## 🔧 Features

- Preprocessing of raw movie reviews
- Dynamic model and vectorizer selection via maps
- Clean CLI output with colored formatting
- Exception handling with traceback output
- Randomized shuffling for reproducibility

---

## 🧠 Dependencies

See `requirements.txt`, or install with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Bash / Linux / macOS

```bash
chmod +x install_and_run.sh
./install_and_run.sh
```

### Windows (PowerShell)

```powershell
.\install_and_run.ps1
```

---

## 📦 Dataset Info

- Downloads `movie_reviews` and `stopwords` from NLTK at runtime
- No need to manually download datasets

---

## 🛠️ Models Used

- Naive Bayes (`MultinomialNB`)
- Support Vector Machine (`LinearSVC`)

---

## 📊 Vectorizers

- CountVectorizer
- TF-IDF Vectorizer with fine-tuned parameters

---

## 📍 Output

Accuracy and classification reports for:
- CountVectorizer + Naive Bayes
- CountVectorizer + SVM
- TF-IDF + Naive Bayes
- TF-IDF + SVM

---

## 🧠 Author

This is a class assignment designed for practicing NLP classification using Python.
