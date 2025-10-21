import nltk
import traceback
import sys
import os
import signal
import re
import numpy as np

from textblob import TextBlob 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

##################### Setup Functions #####################

print("\n\nLoading Program. Please wait...")

# This sets the system colors #
GREEN = '\u001b[92m'
RED = '\u001b[91m'
ORANGE = '\u001b[38;5;208m'
RESET = '\u001b[0m'

def error_msg(e):
    print(f"{RED}Error{RESET}: {ORANGE}{e}{RESET}")
    traceback.print_exc()
    sys.exit(1)

# Terminates the program on Ctrl+C
def sigint_handler(signum, frame):
    print("\nTerminating program...\n")
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)

def shutup():  # This is for when the console is complaining about something stupid
    sys._stderr = sys.stderr  # Backup just once
    sys.stderr = open(os.devnull, 'w')

def restore_sanity():  # This restores error output after being silenced
    if hasattr(sys, '_stderr'):
        sys.stderr = sys._stderr  # Restore
        del sys._stderr
#########################################################

##########Global Properties########################################################################


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

dataset = {
    # greetings / small talk
    r"\b(hi|hello|hey|howdy)\b": "Hi there! How are you?",
    r"\bhow\s+are\s+you\b": "I'm doing well, thanks for asking!",
    r"\b(bye|goodbye|see\s+you)\b": "See you later! Remember to push Q to quit.",

    # identity / capability
    r"\bwhat(?:'s|\s+is)\s+(?:your\s+)?name\b": "I'm a simple chatbot built in Python.",
    r"\bwhat\s+can\s+you\s+do\b": "I can chat with you and answer basic questions.",
    r"\bwho\s+(?:made|created)\s+you\b": "I was created by a programmer just like you.",

    # facts / definitions
    r"\bwhat(?:'s|\s+is)\s+python\b": "Python is a high-level programming language known for its simplicity.",
    r"\bwhat(?:'s|\s+is)\s+ai\b|\bwhat(?:'s|\s+is)\s+artificial\s+intelligence\b": "AI stands for Artificial Intelligence, the simulation of human intelligence by machines.",
    r"\bcapital\b.*\bfrance\b|\bfrance\b.*\bcapital\b": "The capital of France is Paris.",
    r"\b2\s*\+\s*2\b": "2 + 2 = 4",

    # requests / jokes / random
    r"\b(?:tell\s+me\s+a\s+)?jokes?\b": "Why do programmers prefer dark mode? Because light attracts bugs!",
    r"\b(?:sing|sing\s+me\s+a\s+song)\b": "Nope, I don’t have vocal cords.",
    r"\b(?:what(?:'s|\s+is)\s+(?:the\s+)?weather|weather)\b": "I can’t see outside, but you can check a weather app!",
    r"\b(?:tell\s+me\s+something\s+)?cool\b": "Octopuses have three hearts and blue blood.",
    r"\belon\s+musk\b": "Elon Musk is a tech entrepreneur known for Tesla and SpaceX.",

    # personal-ish
    r"\bdo\s+you\s+sleep\b|\bsleep\b": "Not really, I’m always running when you talk to me.",
    r"\bhow\s+old\b": "I don’t age, but my code sure can get outdated.",
    r"\blike\s+me\b|\bdo\s+you\s+like\s+me\b": "I think you’re pretty great to chat with!",

    # time/day
    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?day\b|\bwhat\s+day\s+is\s+it\b": "I don’t have a calendar, but you can check your system clock!",

    # study help
    r"\bstudy\b|\bcan\s+you\s+help\s+me\s+study\b": "Absolutely, just ask me a question and I’ll do my best."
}

alt_responses = [
    "Glad to hear that.",            # 0: positive + objective
    "Love the enthusiasm.",          # 1: positive + subjective
    "Got it, sounds neutral.",       # 2: neutral + objective
    "Alright, noted.",               # 3: neutral + subjective
    "Sounds rough. Want info?",      # 4: negative + objective
    "Excuse me, you can leave!"      # 5: negative + subjective
]


###########################################################################################################################



def data_prep(text):
    try:

        text = text.lower()
        w_tokens = nltk.word_tokenize(text)
        #s_tokens = nltk.sent_tokenize(text)
        sw = set(stopwords.words('english'))
        filtered = [i for i in w_tokens if i.lower() not in sw and i.isalpha()]
        
        return " ".join(filtered)
        
        
    except Exception as e:
        error_msg(e)


def sentiment_eval(text):
    try:

        # Weight matrix (rows = response types)
        W = np.array([
            [  2,  1],   # pos obj
            [  2,  3],   # pos subj
            [  0,  1],   # neutral obj
            [  0,  2],   # neutral subj
            [ -2,  1],   # neg obj
            [ -3,  4]    # neg subj
        ])
        text_eval = TextBlob(text)
        polar = text_eval.sentiment.polarity
        subj = text_eval.sentiment.subjectivity
        matrix = np.array([polar, subj])  

        scores = W.dot(matrix) #matrix multiplication
        idx = np.argmax(scores) #chooses the highest score in matrix and assigns it to the index in the response array
        
        if idx == 5:
            print(alt_responses[idx])
            sys.exit(0)

        return alt_responses[idx]

    except Exception as e:
        error_msg(e)



def input_handle(inquiry):
    try:
        #Important note: Do not query the dataset with cleaned text.
        #Cleaned text has stopwords removed and will result in a negative
        #search result. Use the raw text first, then sent cleaned text to ambigous()

        tokens = data_prep(inquiry)
        raw = inquiry.lower()
        tone_eval = sentiment_eval(raw)
        for pattern, response in dataset.items():
            if re.search(pattern, raw, flags=re.IGNORECASE):
                return response
            
        alternate_response = ambiguous(tokens)
        if alternate_response:
            return tone_eval + " " + alternate_response
        else:
            return tone_eval   

        #return "I did not understand that clearly. Please elaborate?"
        
        
    except Exception as e:
        error_msg(e)


def ambiguous(tokens):
    try:
        #This function evaluation entries that are ambigous or not exact matchies
        input_set = set(tokens.split())
        best_key, best_score = None, 0.0
        
        for key, response in dataset.items():
            #IMPORTANT! Make sure to scrub the regex flags out of the dataset keys, or they WILL get misinterpreted!
            k = re.sub(r'\\[A-Za-z]', ' ', key)  #strips flags such as \b, otherwise they can be misunderstood as escape characters
            k = re.sub(r'\.\*', ' ', k)       #strips *.   
            k = re.sub(r'[^a-z]+', ' ', k.lower()) #this strips non letters out

            key_tokens = set(k.split())            
            
            key_tokens = {t for t in key_tokens if t and t not in stopwords.words('english')}

            if not key_tokens:
                continue
            
            I =  len(input_set & key_tokens) #I is for intersect of the input token set and the key token set
            U = len(input_set | key_tokens) # U is the union
            score =  I / U

            if score > best_score:
                best_key, best_score = key, score

        if best_score >= 0.25:
            return dataset[best_key]
        else:
            return None
    
    except Exception as e:
        error_msg(e)
def main():
    try:
        
        inquiry = input("Good Morning, how can I help you today? (Press Q to quit anytime)  ")
        while True:
            if not inquiry:
                inquiry = input("I'm sorry, I did not get that. Please enter a message: ")
                continue
            response =input_handle(inquiry)
            print(f"{response}")
            inquiry = input()
            if inquiry.lower() == "q":
                print("Goodbye!")
                break
            else:
                continue
        sys.exit(0)


    except Exception as e:
        error_msg(e)


if __name__ == "__main__":
    main()