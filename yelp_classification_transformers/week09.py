import sys
import signal
import traceback
import os

import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn
import requests
from sklearn.preprocessing import LabelEncoder

from transformers import pipeline, AutoTokenizer
import seaborn
import matplotlib
import torch

from datasets import load_dataset, logging
from torch.utils.data import Dataset

from sample_trainer import tokenizer, ds_obj, model_trainer

#####################Packages Installed: #################

'''
hf_xet
datasets
transformers
pandas
matplotlib
seaborn
rich
scikit-learn
'''

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


logging.set_verbosity_error() #This is to quiet all the console chattering when the dataset
                            #is being loaded
def data_prep():
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,  # cleans up after it's done
        ) as progress:

            task = progress.add_task("Loading Yelp dataset...", start=False)
            progress.start_task(task)

            ds = load_dataset("Yelp/yelp_review_full", cache_dir="hf_cache")
            df = ds["train"].to_pandas()

        print("[✓] Done loading dataset.")

        #classification(df)
        return df
    except requests.exceptions.RequestException as e:
        print(f"{RED}Error{RESET}: {ORANGE}{e}{RESET}")
        exit(1)
    except Exception as e:
        error_msg(e)

def classification(frame):
    try:

        new_df = pd.read_csv('samples_cleaned.csv')

        # Encode price_range labels to integers
        label_encoder = LabelEncoder()
        new_df["price_label"] = label_encoder.fit_transform(new_df["price_range"])
        label_classes = list(label_encoder.classes_)
        print("Price range classes:", label_classes)
        #print(new_df)
    except Exception as e:
        error_msg(e)


def main():
    try:
        frame = data_prep()
        classification(frame)

        print('\nTask completed.')
    except Exception as e:
        error_msg(e)

if __name__ == "__main__":
    main()