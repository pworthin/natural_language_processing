import nltk
import sys
import signal
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup



from tqdm import tqdm
from nltk.corpus import reuters, stopwords
from nltk.tokenize import  word_tokenize
from nltk.corpus import words as nltk_words


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

print('\n\nProgram loading. Please wait...')

#### This kills the program if it enters an infinite loop ###

def sigint_handler(signum, frame):
    print("\nTerminating program...\n")
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)


def error_msg(e):
    #This is to generate an error message
    print(f"{RED}Error:{RESET} {e}\n\n")
    traceback.print_exc()
    sys.exit(1)

def shutup(): #This is for when the GUI is complaining about something stupid
    sys._stderr = sys.stderr  # Backup just once
    sys.stderr = open(os.devnull, 'w')

def restore_sanity(): #This restores error output after being silence
    if hasattr(sys, '_stderr'):
        sys.stderr = sys._stderr  # Restore
        del sys._stderr


#############################################################

#This sets the system colors#

GREEN = '\033[92m'
RED = '\033[91m'
ORANGE = '\033[38;5;208m'
RESET = '\033[0m'

#tqdm.pandas(leave=False) #This enables the progress bar for Pandas library




##############################################################


'''
#I'm running into a race condition in data_display()
#where the graphs are being overwritten immediately
#after they are displayed. I am going to store them
#in a global array and see if this works. 
'''
summary_tables = {} 
table_refs = []
stored_figures = [] 
valid_cache = {}
dictionary = set(word.lower() for word in nltk_words.words()) #This sets the dictionary for English words
##############################################################
def data_display(data, vectorizer, method, sw_state):

        try:
            
            df_wordbag = pd.DataFrame(data.toarray(), columns=vectorizer.get_feature_names_out())
            
            word_totals = df_wordbag.sum(axis=0)
            top_words = word_totals.sort_values(ascending=False)
            top_n = 25
            fig, ax = plt.subplots(figsize=(10, 8))
            top_words.head(top_n).plot(kind='barh', color='skyblue', ax=ax)
            ax.set_title(f"Top {top_n} Most Frequent Words in {method} {sw_state}")
            ax.set_xlabel("Frequency")
            ax.invert_yaxis()
            plt.tight_layout()

                
            
            plt.show(block=False)

            stored_figures.append(fig) #This keeps the graph objects in memory so they are not overwritten
            summary_tables[f"{method} {sw_state}"] = df_wordbag #Adding information to our summary table
        except Exception as e:
            error_msg(e)

def valid_check(word, dictionary, sw):
    
    #This is a memoization function to speed the process of checking tokens against
    #the sizeable dictionary
    try:
        key = (word, id(sw))
        if key in valid_cache:
            return valid_cache[key]
        valid = word in dictionary and word not in sw
        valid_cache[key] = valid
        return  valid
    except Exception as e:
         error_msg(e)


def textprep(text, remove_stopwords, stopword_set):

        try:
 
            sw = stopword_set if remove_stopwords else set()
            words_cleaned = re.findall(r'\b[a-z]{4,}\b', text.lower())

            # Remove junky acronyms
            words_cleaned = [word for word in words_cleaned if re.search(r'[aeiou]', word)]

            if remove_stopwords:
                cleaned = [word for word in words_cleaned if valid_check(word, dictionary, sw)]
            else:
                cleaned = words_cleaned

            return ' '.join(cleaned)
            
        except Exception as e:
            error_msg(e)

def analyze_matrices(bow_sw, bow_nosw, tfidf_sw, tfidf_nosw, file_sw):
    
    try:
        
        print(f"\n{GREEN}Comparing Matrix Shapes and Vocab Sizes{RESET}")
        print("-" * 50)
        
        # Matrix shape
        print("BoW w/ SW     shape:", summary_tables["Bag of Words with stopwords"].shape)
        print("BoW w/o SW    shape:", summary_tables["Bag of Words without stopwords"].shape)
        print("TF-IDF w/ SW  shape:", summary_tables["TF-IDF with stopwords"].shape)
        print("TF-IDF w/o SW shape:", summary_tables["TF-IDF without stopwords"].shape)
        

        # Nonzero features per doc
        print("\nAverage # of features per document (non-zero entries):")
        for name in summary_tables:
            mat = summary_tables[name]
            nnz_per_doc = (mat != 0).sum(axis=1)
            avg_features = nnz_per_doc.mean()
            print(f"{name:30}: {avg_features:.2f}")

        # Vocab size difference
        vocab_bow_sw = summary_tables["Bag of Words with stopwords"].shape[1]
        vocab_bow_nosw = summary_tables["Bag of Words without stopwords"].shape[1]
        diff = vocab_bow_sw - vocab_bow_nosw
        print(f"\n{RED}Stopword removal reduced vocab by {diff} words ({(diff/vocab_bow_sw)*100:.1f}%){RESET}")
   
    except Exception as e:
        error_msg(e)

def gui_table():
       
    try:
        import tkinter as tk
        from pandastable import Table
        from tkinter import ttk

        shutup()
        root = tk.Tk()
        root.title("Executive Summary of All Matrices")
        root.geometry("1000x600")

        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True)

        for name, df in summary_tables.items():
            frame = tk.Frame(notebook)
            frame.pack(fill='both', expand=True)
            notebook.add(frame, text=name)

            table = Table(frame, dataframe=df, showtoolbar=False, showstatusbar=False)
            table.show()

        root.mainloop()

    except Exception as e:
        #For some reason, this API keeps complaining when you close the window
        #No idea what it is, but this is going to be the quickfix for now.
         pass  
    finally:
         restore_sanity()


def document_process(filename, stopword_set):
    
    try:
        with open(filename, encoding='utf-8', errors='replace') as f:
                soup = BeautifulSoup(f, "html.parser")

    # Extract visible text from the document file
                text = soup.get_text(separator=' ', strip=True)
                print(text[:2000])
                cleaned_file =textprep(text, True, stopword_set)
        
        return [cleaned_file]
    
    
    except Exception as e:
        error_msg(e)

def thread_processor(data, stopword_set):
     
     try:
            with ThreadPoolExecutor(max_workers=2) as executor:
            #I had to implement a multithreading function here because of the enormous size
            #of the datasets.
                preprocess_corpus_sw = list(tqdm(
                    executor.map(lambda t: textprep(t, False, stopword_set), data),
                    total=len(data),
                    desc="Cleaning Corpus w/ stopwords...",
                    ncols=100
                ))
                preprocess_corpus_nosw = list(tqdm(
                    executor.map(lambda t: textprep(t, True, stopword_set), data),
                    total=len(data),
                    desc="Cleaning Corpus w/o stopwords...",
                    ncols=100
                )) 
            
            return preprocess_corpus_nosw, preprocess_corpus_sw
     
     except Exception as e:
            error_msg(e)
      

class NLPModel:

    def __init__(self, text, type, stopwords, label=None, **params):
        self.stopwords = stopwords
        self.type = type
        self.text = text
        self.params = params
        self.label = label

        if len(text) == 1:
            self.params.setdefault("min_df", 1)
            self.params.setdefault("max_df", 1.0)
            
          
    def wordbag(self):
        #This function creates a Bag of Words representation using
        #CountVectorizer()
        '''
        Special Note: CountVectorizer() does not appear to catch all
        stopwords, which causes the display to spit out garbage tokens.
        Parameters have been added to fine tune the cleaned text.
        '''

        #Update: Adding TF-IDF method TfidVectorizer() to save space. I will include
        #a function to prompt 1 for Bag of Words and 2 of TF-IDF
        try:
            if self.type == 'bow':
            
                vectorizer = CountVectorizer(**self.params)
                method = "Bag of Words"
                
            elif self.type == 'tfidf':

                    vectorizer = TfidfVectorizer(**self.params)
                    
                    method = "TF-IDF"

            if self.stopwords:
                    sw_state = "with stopwords"
            elif self.label:
                    sw_state = f"without stopwords ({self.label})"
            else:
                 sw_state = "without stopwords"
            
            matrix = vectorizer.fit_transform(self.text)

            print(f"\n\n{GREEN}Vocabulary size for {method} {sw_state}{RESET}:", len(vectorizer.vocabulary_))
            print(f"{GREEN}{method} Matrix shape (Documents processed){RESET}:", matrix.shape)
            print("\n")
            for word, index in list(vectorizer.vocabulary_.items())[:10]:
                print(f"{word}: {ORANGE}{index}{RESET}")
            print("-" * 60)
            print("\n\n")
            data_display(matrix, vectorizer, method, sw_state)

        except Exception as e:
            error_msg(e)



        
def main():

    
    
    try:
        nltk.download('reuters', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('stopwords', quiet=True)
        stopword_set = set(stopwords.words('english'))  

        
        if not reuters.fileids():
            print("Reuters corpus not loaded.")
            return
        
        documents = reuters.fileids()
        corpus = [reuters.raw(doc_id) for doc_id in documents]
        rawfile = document_process("crypto.html", stopword_set)
        
        
        print(f"Loading {len(corpus)} documents")
        #print(f"{corpus[:1]}\n\n")

        
        
       
        preprocess_corpus_nosw, preprocess_corpus_sw = thread_processor(corpus, stopword_set)
            
        file_nosw =NLPModel(rawfile, 'tfidf', False, 'crypto.html')
        bow_sw = NLPModel(preprocess_corpus_sw, 'bow', True, None, max_df=0.85, min_df=5)
        bow_nosw = NLPModel(preprocess_corpus_nosw, 'bow', False, None, max_df=0.85, min_df=5)
        tfidf_nosw = NLPModel(preprocess_corpus_nosw, 'tfidf', False, None, max_df=0.85, min_df=5)
        tfidf_sw = NLPModel(preprocess_corpus_sw, 'tfidf', True, None, max_df=0.85, min_df=5)

        models = [bow_sw, bow_nosw, tfidf_nosw, tfidf_sw, file_nosw]

        for model in models:
                model.wordbag()

        print(f"{RED}Note{RESET}: Graph will stay open. Close it manually when done.")
        gui_table()
                                
        input("\nPress Enter to exit and close all graphs...")
        analyze_matrices(bow_sw, bow_nosw, tfidf_sw, tfidf_nosw, file_nosw)
        print("Program successfully completed. Terminating...")
        sys.exit(0)

    except Exception as e:
        error_msg(e)


if __name__ == "__main__":
    main()
    