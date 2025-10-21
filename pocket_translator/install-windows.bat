@echo off
echo [*] Creating virtual environment...
python -m venv venv

echo [*] Activating virtual environment...
call venv\Scripts\activate

echo [*] Installing requirements...
pip install -r requirements.txt

echo [*] Running translator...
python Week07v1.py

pause
