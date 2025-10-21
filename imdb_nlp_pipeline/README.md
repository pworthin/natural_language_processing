# IMDB Review Processing and Vectorization

This repo showcases text processing workflows using NLTK and spaCy on IMDB movie reviews. These are split across two Python scripts:

- **Week1.py**: Basic text preprocessing, tokenization, and Bag-of-Words vectorization using NLTK.
- **Week2.py**: Extended NLP operations including spaCy comparisons, n-gram vectorization, model training with Logistic Regression and Naive Bayes.

---

## 📦 Dependencies

Install with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Bash / Linux / macOS

```bash
chmod +x install_and_run_week1_2.sh
./install_and_run_week1_2.sh
```

### Windows (PowerShell)

```powershell
.\install_and_run_week1_2.ps1
```

---

## 📊 Output

- `movie_reviews.csv` — saved IMDB data (Week 1 & 2)
- `processed_reviews_comparison.csv` — tokenization/cleaning comparison (Week 1)
- `wordbag_representation.csv` — vectorized bag of words output (Week 1)
- Console output of token stats, n-gram features, and model performance

