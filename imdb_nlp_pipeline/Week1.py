import datasets
import pandas as pd
import nltk
import traceback
import numpy as np
import signal
import sys

from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize
from nltk.tokenize import sent_tokenize
from tqdm import tqdm




from sklearn.feature_extraction.text import CountVectorizer

#### This kills the program if it enters an infinite loop ###

def sigint_handler(signum, frame):
    print("\nTerminating program...\n")
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)

#############################################################

#This sets the system colors#

GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

tqdm.pandas(leave=False) #This enables the progress bar for Pandas library

##############################################################


def data_prep():

    # This function retrieves the databset IMDB, renames the columns, and saves the information to disk.

    try:

        print("\nBuilding dataframe...", end='', flush=True)
        ds = datasets.load_dataset('imdb')
        frame = pd.DataFrame(ds['train'])
        frame.rename(columns={'text':'review_text', 'label':'sentiment'}, inplace=True)
        frame['sentiment'] = frame['sentiment'].map({0: 'negative', 1: 'positive'})
        print(f"{GREEN}Successfull!{RESET}")
        print("Saving 'movie_reviews.csv'...", end='', flush=True)
        frame.to_csv('movie_reviews.csv', index=False)
        print(f"{GREEN}Successfull!{RESET}")
        print("Would you like to view the table so far? (y/n)  ", end='', flush=True)
        response = input()
        if(response == 'y'):
            print(frame)

        return frame

    except Exception as e:

        print(f"{RED}Error:{RESET} {e}\n\n")
        traceback.print_exc()
        sys.exit(1)
        


def text_prep(text):  # Expects a single review string--do NOT pass in the entire dataframe!
                    #   It will throw an error!
   
    
    try:       

        tokens = word_tokenize(text.lower())
        sw = set(stopwords.words('english'))
        filtered = []

        for token in tokens:
            if token.isalpha():
                if token not in sw:
                    filtered.append(token)
        
        return ' '.join(filtered)

    except Exception as e:
        print(f"{RED}Error:{RESET} {e}\n\n")
        traceback.print_exc()
        sys.exit(1)

def tokenization(text):
    
        #This function tokenizes words and sentences
    try:
        
        
        sentences = sent_tokenize(text, language='english')
        words = word_tokenize(text)
        
        return pd.Series([sentences, words])  # returns both in one call

    except Exception as e:
        print(f"{RED}Error:{RESET} {e}\n\n")
        traceback.print_exc()
        sys.exit(1)
    


def vectorization(frame):

    try:
        
        v = CountVectorizer(max_features=1000)
        wordbag_matrix = v.fit_transform(frame['cleaned'])  # use cleaned text!

        # Convert sparse matrix to dataframe for saving and analysis
        wordbag = pd.DataFrame(wordbag_matrix.toarray(), columns=v.get_feature_names_out())

        word_freq = np.sum(wordbag.values, axis=0)
        vocab = wordbag.columns

        # Named inner function to extract frequency for sorting
        def get_frequency(pair):
            return pair[1]  # pair is (word, frequency)

        # Combine vocab and frequency
        word_freq_pairs = list(zip(vocab, word_freq))

        # Sort by frequency in descending order
        sorted_pairs = sorted(word_freq_pairs, key=get_frequency, reverse=True)

        # Grab top 20
        top_words = sorted_pairs[:20]
        print(f"{GREEN}Successfull!{RESET}")
        for word, freq in top_words:
            print(f"{word}: {freq}")

        return wordbag

    except Exception as e:
        print(f"{RED}Error:{RESET} {e}\n\n")
        traceback.print_exc()
        sys.exit(1)

def main():

    try:
        
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

        frame = data_prep()
        print("\nPreparing text for tokenization...")
        frame['cleaned'] = frame['review_text'].progress_apply(text_prep)
        print(f"{GREEN}Successfull!{RESET}")
        print("Tokenizing sentences and words...")
        frame[['sent_tokenized', 'word_tokenized']] = frame['review_text'].progress_apply(tokenization)
        print(f"{GREEN}Successfull!{RESET}")
        print("Vectorizing data...", end='', flush=True)
        wordbag = vectorization(frame)

        # Add comparison columns
        frame['num_sentences'] = frame['sent_tokenized'].progress_apply(len)
        frame['words_before'] = frame['word_tokenized'].progress_apply(len)
        frame['words_after'] = frame['cleaned'].progress_apply(lambda x: len(x.split()))
        
        print("\nPlease wait....\n")

        # Create comparison DataFrame
        comparison = frame[['review_text', 'cleaned', 'num_sentences', 'words_before', 'words_after']]
        comparison.to_csv('processed_reviews_comparison.csv', index=False)
        print("Saving comparison file to 'processed_revies_comparison.csv'...\n")

        # Print average stats
        avg_sentences = comparison['num_sentences'].mean()
        avg_words_before = comparison['words_before'].mean()
        avg_words_after = comparison['words_after'].mean()

        print("Review Stats (Averages):")
        print(f"- Avg sentences per review: {avg_sentences:.2f}")
        print(f"- Avg word count before preprocessing: {avg_words_before:.2f}")
        print(f"- Avg word count after preprocessing: {avg_words_after:.2f}")

        #Show first 5 rows of comparison
        print("Sample comparison data:")
        print(comparison.head())

        # Save bag of words matrix (already done in vectorization step, but let's be explicit)
        wordbag.to_csv('wordbag_representation.csv', index=False)

        

    except Exception as e:
        print(f"{RED}Error:{RESET} {e}\n\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
