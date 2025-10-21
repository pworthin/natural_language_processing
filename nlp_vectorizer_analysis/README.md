# NLP Vectorizer Analysis with Reuters Dataset

This script processes the Reuters corpus and a custom HTML file (`crypto.html`) to demonstrate the impact of stopword removal and vectorization methods (BoW and TF-IDF). It includes:

- Text cleaning and preprocessing
- Optional stopword removal
- Word frequency visualizations
- Matrix shape and vocabulary comparison
- GUI table output using `tkinter` and `pandastable`

---

## 🧠 Dependencies

Install them via pip:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Bash / Linux / macOS

```bash
chmod +x install_and_run_week3.sh
./install_and_run_week3.sh
```

### Windows (PowerShell)

```powershell
.\install_and_run_week3.ps1
```

---

## 📂 Input File

Make sure you have a file named `crypto.html` in the same directory. This file will be processed and vectorized alongside the Reuters corpus.

---

## 📊 Output

- Four graphs showing the top 25 most frequent words per configuration
- GUI summary of vectorized matrices
- Terminal comparison of shapes and vocabulary reductions

