@echo off
echo Installing required Python packages...
pip install -r requirements.txt

echo Downloading necessary NLTK datasets...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo Setup complete. You can now run week06.py
pause
