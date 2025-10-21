# Yelp Review Classification using HuggingFace Transformers

This project demonstrates how to classify Yelp reviews using the HuggingFace Transformers ecosystem. It uses the `Yelp Review Full` dataset and includes:

- Dataset loading via `datasets` library
- Label encoding for custom classification
- External module hooks for training and tokenization
- Terminal-friendly output using `rich`

---

## 📦 Dependencies

Install required packages with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Bash / Linux / macOS

```bash
chmod +x install_and_run_week9.sh
./install_and_run_week9.sh
```

### Windows (PowerShell)

```powershell
.\install_and_run_week9.ps1
```

---

## 🧠 Notes

- `sample_trainer.py` is a required companion script that must define:
  - `tokenizer`
  - `ds_obj`
  - `model_trainer`
- `samples_cleaned.csv` is expected to be in the same directory if running the `classification()` function.
- Uses `cache_dir="hf_cache"` to locally cache HuggingFace datasets

---

## 📁 Structure

- `week09.py` – main entry point
- `sample_trainer.py` – define training logic and tokenizer
- `samples_cleaned.csv` – expected CSV input for classification
