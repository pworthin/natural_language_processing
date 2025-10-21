Write-Host "Creating virtual environment..."
python -m venv venv
.env\Scripts\Activate.ps1

Write-Host "Installing requirements..."
pip install -r requirements.txt

Write-Host "Running program..."
python week04.py
