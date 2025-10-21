
import sys
import signal
import traceback
import os
import json
import re


from rich.console import Console
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor



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

def graph_stats(scores, filename):
    try:
        #Note: Matplotlib does not clear its memory after executing.
        #Remember to include this two functions to flush the plt object
        plt.figure()

        compounds = [s['compound'] for s in scores]

        plt.hist(compounds, bins=30, edgecolor='black')
        plt.title(f"Sentiment Score Distribution for file {filename}")
        plt.xlabel("Compound Score")
        plt.ylabel("Number of Tweets")
        plt.grid(True)
        plt.tight_layout()
        plt.ion()
        plt.show()
        plt.pause(1)

    except Exception as e:
        error_msg(e)

def top_word_freqs(text, stopwords, n=50):
    try:
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        counts = Counter(tokens)

        return dict(counts.most_common(n))

    except Exception as e:
        error_msg(e)


class CloudBuild:
    def __init__(self, width=800, height=400):
        self.width = width
        self.height = height

    def generate_cloud(self, text, **kwargs):
        try:
            wordcloudbuild = WordCloud(
                width=self.width,
                height=self.height,
                background_color='white',
                #colormap=colormap,
                max_words=200,
                #mask=mask,
                contour_width=3,
                relative_scaling=0.25,
                collocations=False, #This is to avoid it from joining bigrams
                contour_color='steelblue',
                **kwargs
            )
            if isinstance(text, dict):
                wordcloud = wordcloudbuild.generate_from_frequencies(text)
            else:
                wordcloud = wordcloudbuild.generate(text)

            return wordcloud

        except Exception as e:
            error_msg(e)



    def plot_cloud(self, wordcloud, title="Word Cloud Visualization"):
        try:
            plt.figure(figsize=(15, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title)
            plt.ion()
            plt.show()
            plt.pause(1)
            #input("Press any key to continue...")
        except Exception as e:
            error_msg(e)

    def cloud_create(self, text, stopwords):
        try:
            with open(text, 'r') as file:
                text_input = file.read()
            freqs = top_word_freqs(text_input, stopwords, n=50)
            print(f"Top Frequent 50 Words:\n")
            for i in freqs:
                print(i)

            cloud_test = CloudBuild()
            raw_cloud = cloud_test.generate_cloud(text_input, stopwords=stopwords)
            cloud = cloud_test.generate_cloud(freqs)
            cloud_test.plot_cloud(raw_cloud, "Original Cloud, no stopwords removed")
            cloud_test.plot_cloud(cloud, "Clean Text Cloud")

        except Exception as e:
            error_msg(e)

def thread_processor(tweets, analyzer, chunk_size=100):
    #VADER is UNBEARABLY SLOW! MY GOD! Here is a multi-threading
    #function to break it into batches and speed the process.
    #Note: This function was AI generated, as I am still clumsy with
    #multithread programming and right now don't have time to play around with it.
    try:

        def chunked(data, size):
            for i in range(0, len(data), size):
                yield data[i:i + size]

        def analyze_chunk(chunk):
            return [analyzer.polarity_scores(tweet) for tweet in chunk]

        chunks = list(chunked(tweets, chunk_size))
        all_scores = []

        with ThreadPoolExecutor() as executor:
            results = executor.map(analyze_chunk, chunks)
            for result in results:
                all_scores.extend(result)

        return all_scores

    except Exception as e:
        error_msg(e)

class SentimentAnalyzer:
    def __init__(self, text):

        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.text = text

    def analyze(self):
        try:
            #These are JSON files, they need to be handled differently!
            with open(self.text, 'r') as file:
                neg_tweets = [json.loads(j) for j in file]
            if not neg_tweets:
                raise ValueError("No JSON text found")


            #There is a lot of noise in these files. I had to makd another function to clean them
            #I should also note someone in these tweets has an unhealthy obsession with Justin Bieber...

            blacklist = ['follower', 'follow', 'unfollower', 'rt']

            raw_tweets = list(set([ # Make sure to use the set function to knock out duplicate tweets
                self.clean(i['text'])
                for i in neg_tweets
                if 'text' in i
                   and i['text'].strip()
                   and not any(word in i['text'].lower() for word in blacklist)
            ]))



            print(f"\nLoaded {len(raw_tweets)} tweets from {self.text} for sentiment analysis.")

            all_scores = thread_processor(raw_tweets, self.vader_analyzer)
            aggregate_scores.append(all_scores)
            extreme_neg = []
            extreme_pos = []

            for tweet, score in zip(raw_tweets, all_scores):
                score['text'] = tweet
                if score['compound'] <= -0.08:
                    extreme_neg.append(score)
                elif score['compound'] >= 0.8:
                    extreme_pos.append(score)
             #!!!IMPORTANT!!! If there is an entry without a key, the program will throw a KeyError
            #Mitigate it by using the .get() function with a 0 value to let the interpreter know not to
            #expect a value
            extreme_neg = sorted(extreme_neg, key=lambda x: x.get('compound', 0))
            extreme_pos = sorted(extreme_pos, key=lambda x: x.get('compound', 0)) #I need to remember to start using lambda
                                                                                  #methods, not basic helper functions
                                                                                 #that clutter the parent functions. Lambda!
            #I am going to call the histogram function from here
            graph_stats(all_scores, self.text)

            print(f"\nNegative Tweets from {self.text}: ")
            for t in extreme_neg:
                print(f"{t['compound']:.2f}\t{t['text']}")

            print(f"\nExtremely Positive Tweets from {self.text}")
            for t in extreme_pos:
                print(f"{t['compound']:.2f}\t{t['text']}")


        except Exception as e:
            error_msg(e)

            # This is a private helper function to clean out the noise in the twitter files
    def clean(self, text):
        try:
            text = text.replace('\n', ' ').replace('\r', ' ')  # strip newlines
            text = text.lower()
            text = re.sub(r"http\S+", " ", text)  # kill URLs
            text = re.sub(r"\b(?:t|c|x|d|p|lt)\b", " ", text)  # remove standalone t, c, u, x
            text = re.sub(r"[^a-z\s]", " ", text)  # only letters + spaces
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            error_msg(e)


aggregate_scores = []
def main():
    try:
        stopwords = set(STOPWORDS)
        stopwords.update({'involves', 'day', 'oil', 'make', 'often', 'might', 'may', 'using',
                          'used', 'allow', 'without'})
        cloud = CloudBuild()
        cloud.cloud_create('sampletext.txt', stopwords)



        tweets = ['negative_tweets.json', 'positive_tweets.json', 'tweets.20150430-223406.json']

        for i in tweets:
            obj = SentimentAnalyzer(i)
            obj.analyze()
        flat_scores = [s for sublist in aggregate_scores for s in sublist]
        graph_stats(flat_scores, '[Aggregate Scores]')
        input("\nPress any key to terminate program...")


        ''' Don't mind this, this is a test script I'm referencing to
            and I will probably forget to delete it...
            
        print(f"\n\n{'Rating':<20} {'Score':>7}")
        print("-" * 29)
        for i, j in neg_summary.items():
            print(f"{i:<20}{j:>7.2f}")
        '''

    except Exception as e:
        error_msg(e)


if __name__ == '__main__':
    main()