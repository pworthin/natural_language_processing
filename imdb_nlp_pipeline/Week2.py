print("Initiating program, loading necessary files. Please wait...")

import pandas as pd
import nltk
import spacy
import traceback
import numpy as np
import signal
import sys

from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize
from nltk.tokenize import sent_tokenize
from tqdm import tqdm




from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


'''Special note: This program takes an object oriented approach to creating and comparing
   models, as it would be more practical as opposed to recreating the same functions
   needlessly. The new functions for this assignment will be appropriately noted.


'''

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
#############################################################

#This sets the system colors#

GREEN = '\033[92m'
RED = '\033[91m'
ORANGE = '\033[38;5;208m'
RESET = '\033[0m'

tqdm.pandas(leave=False) #This enables the progress bar for Pandas library

def error_msg(e):
    #This is to generate an error message
    print(f"{RED}Error:{RESET} {e}\n\n")
    traceback.print_exc()
    sys.exit(1)
##############################################################


########################Added functions for Week 2 assignment##########################


'''This object loads a spaCy model from its library.
   Due to the structure of the program, I unfortunately
   was forced to make it global in order to create a separate
   text preparation function for it.
'''

spacy_nlp = spacy.load("en_core_web_sm")



class NLPModel:
   
   #This class constructs an object to represent the type of model processing the text
   
    def __init__(self, name, rawtxt=None, cleantxt=None, tokSent=None, tokWords=None):
      
        self.name = name
        self.rawtxt = rawtxt
        self.cleantxt = cleantxt
        self.tokSent = tokSent
        self.tokWords = tokWords

    def preprocess_nltk(self):


        try:
            self.cleantxt = nltk_txt(self.rawtxt)
            self.lctokens = word_tokenize(self.rawtxt.lower()) #Tokenizes and lowercase token

            sw = set(stopwords.words('english'))
            self.nonsw = [] #this array holds the non stopwords found

            for token in self.lctokens:
                if token.isalpha() and token not in sw:
                    self.nonsw.append(token)

        except Exception as e:
            error_msg(e)
    
    
    def preprocess_spacy(self):
        
        
        try:
            self.cleantxt = spacytxt(self.rawtxt)

            # Tokenize and lowercase with the spaCy function
            doc = spacy_nlp(self.rawtxt.lower())

            # Capture tokens as-is
            self.lctokens = []
            for token in doc:
                self.lctokens.append(token.text)

            # Now remove stopwords and non-alphabetical tokens
            self.nonsw = []
            for token in doc:
                if token.is_alpha and not token.is_stop:
                    self.nonsw.append(token.text)

        except Exception as e:
            error_msg(e)

    def tokenize(self):
        try:
            self.tokSent, self.tokWords = tokenization(self.rawtxt)
        except Exception as e:
        
            error_msg(e)

    def model_cmp(self):

            #Here we are comparing NLTK with spaCy
        try:

            print(f"{RED}--- {self.name} ---{RESET}")
            print(f"Raw Text: {self.rawtxt}\n")

            # NLTK Processing
            print(f"{ORANGE}[NLTK Preprocessing]{RESET}")
            self.preprocess_nltk()
            print(f"Lowercased / Tokenized: {self.lctokens}")
            print(f"After Stopword Removal: {self.nonsw}")
            print(f"Final Cleaned Text: {self.cleantxt}")
            print(f"Token Count: {len(self.lctokens)} -> {len(self.nonsw)}\n")

            # spaCy Processing
            print(f"{ORANGE}[spaCy Preprocessing]{RESET}")
            self.preprocess_spacy()
            print(f"Lowercased / Tokenized: {self.lctokens}")
            print(f"After Stopword Removal: {self.nonsw}")
            print(f"Final Cleaned Text: {self.cleantxt}")
            print(f"Token Count: {len(self.lctokens)} -> {len(self.nonsw)}")

            print("-" * 80)

        except Exception as e:
            error_msg(e)


    
    #This here is just an object test function to make sure it's working
    def __repr__(self):
        return f"<NLPModel(name={self.name})>"    





def sample_cmp():
    #This is our test function for Week 2 to compare the models
    try:
        reviews = [
            {"text": "I'm not sure what I expected... but *that* ending? Just wow.", "label": 1},
            {"text": "It's like Tarantino met David Lynch at a peyote rave.", "label": 1},
            {"text": "Worst. Movie. Ever. 90 mins I'll never get back.", "label": 0},
            {"text": "She said, “Don't go in there,” and I screamed: 'RUN!'", "label": 0},
            {"text": "Even with 4K resolution, the CGI looked like a PS2 cutscene.", "label": 0}
                ]   


        print(f"{GREEN}Comparing NLTK vs spaCy on sample reviews:{RESET}\n")

        for i, entry in enumerate(reviews):
            review_obj = NLPModel(name=f"Review_{i+1}", rawtxt=entry["text"])
            review_obj.model_cmp()


    except Exception as e:
        error_msg(e)

def getTxt():

    #For now this is going to be a helper function to work with the different vectorizers
    #I'm probably going to use this function in the main class if I can get it to cooperate...

    

    """
    Preprocesses 5 sample reviews using spaCy and returns cleaned review text after lowercasing & stopword removal
    and the sentiment labels (1 = positive, 0 = negative)

    You can switch from spaCy to NLTK here if needed by calling obj.preprocess_nltk() instead. Both have been 
    tested
    """

    try:
        reviews = [
            {"text": "I'm not sure what I expected... but *that* ending? Just wow.", "label": 1},
            {"text": "It's like Tarantino met David Lynch at a peyote rave.", "label": 1},
            {"text": "Worst. Movie. Ever. 90 mins I'll never get back.", "label": 0},
            {"text": "She said, “Don't go in there,” and I screamed: 'RUN!'", "label": 0},
            {"text": "Even with 4K resolution, the CGI looked like a PS2 cutscene.", "label": 0}
        ]

        cleaned_reviews = []
        labels = []

        for i, entry in enumerate(reviews):

            obj = NLPModel(name=f"Review_{i+1}", rawtxt=entry["text"])
            obj.preprocess_spacy()  # I'm going to switch to NLTK later for comparison
            cleaned_reviews.append(obj.cleantxt)
            labels.append(entry["label"])

        return cleaned_reviews, labels
    

    except Exception as e:

        error_msg(e)



def vector_cmp(texts):

    #This is where I'll be comparing the Count vs Tfid Vectorizer.
    #For some reason the vocabulary size between the two is always
    #the same and I am not understanding why.

    try:

        print(f"{GREEN}Vectorizing with CountVectorizer...{RESET}")
        count_vec = CountVectorizer()
        count_matrix = count_vec.fit_transform(texts)
        count_features = count_vec.get_feature_names_out()
        print(f"- Vocabulary Size: {len(count_features)}")
        print(f"- Sample Features: {count_features[:10]}")

        print(f"\n{GREEN}Vectorizing with TfidfVectorizer...{RESET}")
        tfidf_vec = TfidfVectorizer()
        tfidf_matrix = tfidf_vec.fit_transform(texts)
        tfidf_features = tfidf_vec.get_feature_names_out()
        print(f"- Vocabulary Size: {len(tfidf_features)}")
        print(f"- Sample Features: {tfidf_features[:10]}")


        return count_vec, tfidf_vec, count_matrix, tfidf_matrix

    except Exception as e:

         error_msg(e)


def ngEval(texts):


    try:
            
        print(f"{GREEN}Analyzing n-gram ranges...{RESET}")

        ranges = [(1, 1), (1, 2), (2, 2)]

        for ngram_range in ranges:


            print(f"\n{ORANGE}N-gram Range: {ngram_range}{RESET}")
            
            # Count Vectorizer
            count_vec = CountVectorizer(ngram_range=ngram_range)
            count_matrix = count_vec.fit_transform(texts)
            count_features = count_vec.get_feature_names_out()
            print(f"[CountVectorizer] Feature count: {len(count_features)}")
            print(f"  Sample features: {count_features[:10]}")

            # TF-IDF Vectorizer
            tfidf_vec = TfidfVectorizer(ngram_range=ngram_range)
            tfidf_matrix = tfidf_vec.fit_transform(texts)
            tfidf_features = tfidf_vec.get_feature_names_out()
            print(f"[TfidfVectorizer] Feature count: {len(tfidf_features)}")
            print(f"  Sample features: {tfidf_features[:10]}")

    except Exception as e:
        
        error_msg(e)        



def spacytxt(text):

    #This added class is for preparing text analyzed with the spaCy library

    try:
        doc = spacy_nlp(text.lower())

        filtered = []
        for token in doc:
            if token.is_alpha and not token.is_stop:
                filtered.append(token.text)

        return ' '.join(filtered)


    
    except Exception as e:
        error_msg(e)



def topterms(tfidf_matrix, tfidf_vectorizer, top_n=5):

    """
This is to display the top n terms with the highest TF-IDF scores for each document (review).

Args:
- tfidf_matrix: sparse matrix output from TfidfVectorizer
- tfidf_vectorizer: TfidfVectorizer object
- top_n: number of top terms to show per review


"""


    try:


        feature_names = tfidf_vectorizer.get_feature_names_out()

        print(f"\n{GREEN}Top TF-IDF Terms Per Document:{RESET}")

        for doc_index in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[doc_index].toarray()[0]
            scores = list(zip(feature_names, row))
            srtscores = sorted(scores, key=lambda x: x[1], reverse=True)

        top_terms = []
        
        for term, score in srtscores[:top_n]:
            rndscore = round(score, 4)
            top_terms.append((term, rndscore))
            
            print(f"\n{ORANGE}Review {doc_index + 1}:{RESET}")
            
            for term, score in top_terms:
                print(f"  {term}: {score}")
    
    
    except Exception as e:
        error_msg(e)


def rating_class(texts, labels, use_tfidf=True):

    """
    Train/split and classification using either CountVectorizer or TfidfVectorizer.
    Prints accuracy and prediction report.

    """

    try:
        # Choose vectorizer
        vec = TfidfVectorizer() if use_tfidf else CountVectorizer()
        X = vec.fit_transform(texts)
        y = labels

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        # Logistic Regression
        print(f"\n{ORANGE}Logistic Regression Results:{RESET}")
        log_clf = LogisticRegression()
        log_clf.fit(X_train, y_train)
        y_pred_log = log_clf.predict(X_test)
        print(classification_report(y_test, y_pred_log, zero_division=0))
        print(f"Accuracy: {round(accuracy_score(y_test, y_pred_log) * 100, 2)}%")

        #  Naive Bayes
        print(f"\n{ORANGE}Naive Bayes Results:{RESET}")
        nb_clf = MultinomialNB()
        nb_clf.fit(X_train, y_train)
        y_pred_nb = nb_clf.predict(X_test)
        print(classification_report(y_test, y_pred_nb, zero_division=0))
        print(f"Accuracy: {round(accuracy_score(y_test, y_pred_nb) * 100, 2)}%")
    
    
    except Exception as e:
       
        error_msg(e)

    


############## End of Added Functions for Week 2 ########################

def data_prep():

    # This function retrieves the databset IMDB, renames the columns, and saves the information to disk.

    try:

        print("\nBuilding dataframe...", end='', flush=True)
        import datasets
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

        error_msg(e)
        


def nltk_txt(text):  

   '''
    The name of this function is changed from "text_prep" to "nltk_txt"
    to distinguish from the spaCy text prep function. 
    This prepare text that will be analyzed with the NLTK library
    Expects a single review string--do NOT pass in the entire dataframe!
    It will throw an error!
                    
    '''
   
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
        error_msg(e)




def tokenization(text):
    
        #This function tokenizes words and sentences
    try:
        
        
        sentences = sent_tokenize(text, language='english')
        words = word_tokenize(text)
        
        return pd.Series([sentences, words])  # returns both in one call

    except Exception as e:
        error_msg(e)
    


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
        error_msg(e)

def main():
    
    try:
        
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

        sample_cmp()
        texts, labels = getTxt()
        vector_cmp(texts)
        ngEval(texts)        
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        topterms(tfidf_matrix, tfidf_vectorizer, top_n=5)
        rating_class(texts, labels, use_tfidf=True)

    

    except Exception as e:
        error_msg(e)


if __name__ == "__main__":
    main()
