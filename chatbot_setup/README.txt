Chatbot Setup Guide
===================

This is a simple rule-based chatbot enhanced with basic sentiment analysis.
To get started, follow the steps for your operating system below:

-------------------------
Windows Users:
-------------------------
1. Double-click install.bat to install dependencies and download NLTK data.
2. Run the chatbot:
   > python week06.py

-------------------------
Linux/Mac Users:
-------------------------
1. Open a terminal and run:
   > chmod +x install.sh
   > ./install.sh

2. Launch the chatbot:
   > python3 week06.py

-------------------------
Features:
-------------------------
- Regex pattern-matching for common phrases and questions
- Sentiment evaluation using TextBlob and a weighted matrix
- Tokenization and stopword filtering via NLTK
- Ambiguous input resolution via Jaccard similarity

-------------------------
Dependencies:
-------------------------
- nltk
- textblob
- numpy

-------------------------
Note:
-------------------------
You can quit the chatbot any time by typing: Q

Enjoy chatting!
