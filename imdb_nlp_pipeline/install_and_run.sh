#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

echo "Running Week 1 script..."
python Week1.py

echo "Running Week 2 script..."
python Week2.py
