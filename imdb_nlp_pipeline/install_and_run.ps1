Write-Host "Creating virtual environment..."
python -m venv venv
.env\Scripts\Activate.ps1

Write-Host "Installing requirements..."
pip install -r requirements.txt

Write-Host "Running Week 1 script..."
python Week1.py

Write-Host "Running Week 2 script..."
python Week2.py
