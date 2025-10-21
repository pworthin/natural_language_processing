import traceback
import sys
import os
import signal

import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tabulate import tabulate
#from datasets import load_dataset

from sklearn.metrics import classification_report



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



def model_setup(model):
    try:
         with open('tech_news_mashup.txt') as file:
            text= file.read()    


            document = model(text)

            #for i in document.ents:
             #   print(f"\nText: {i.text: <30} \tLabel: {i.label_} \t\t\t\tDescription: {spacy.explain(i.label_)}")
            data = [(ent.text, ent.label_, spacy.explain(ent.label_)) for ent in document.ents]
            print(tabulate(data, headers=["Text", "Entity", "Description"], tablefmt="grid", colalign=("left", "left", "left")))
            eval(data)
            data_analysis(data)
            
    except Exception as e:
        error_msg(e)








def graph_setup(ent_total, case):
    try:

        def graph_params():
        #I like calling a separate helper function with a dictionary of parameters
        #They are easier to adjust
            try:
                params = {

                    "bar_width": 0.5,
                    "y_step": 20,
                    "figsize": [12,5],
                    "rotation": 45,
                    "threshold": 15
                }

                return params
            except Exception as e:
                error_msg(e)

        params = graph_params()
        
        plt.ion() #IMPORTANT! If this is not called, the graph either stall the program
                  #Or immediately vanish due to a race condition w/ the garbage collector
        
        
        plt.figure(figsize=params["figsize"])
        ax = sns.barplot(x=ent_total.index, y=ent_total.values, width=params["bar_width"])  

        # y-axis ticks every y_step
        ymax = int(ent_total.max())
        params["y_step"] = 15 if ymax < 200 else params["y_step"]
        ax.set_yticks(np.arange(0, ymax + params["y_step"], params["y_step"]))

        plt.xticks(rotation=params["rotation"])
        plt.title(f"Entity Distribution by Label{case}")
        plt.xlabel("Entity Type")
        plt.ylabel("Count")
        ax.yaxis.grid(True, linestyle=":", alpha=0.9)  # Sets horizontal lines in the background
        ax.set_axisbelow(True)  # Put the grid behind the bars
        plt.tight_layout()
        plt.show(block=False)
        
        print(ent_total, "\n\n")



    except Exception as e:
        error_msg(e)


def load_gold_pairs(path, labels=None):
    #This is an AI-generated repair function I had to implement after the evaluation function 
    #somehow become confused reading the contents of the external JSON
    #file.
    import json, re
    def norm(s):
        s = s.lower()
        s = s.replace("’","'").replace("“",'"').replace("”",'"').replace("–","-").replace("—","-")
        s = re.sub(r"\s+", " ", s.strip())
        return s

    data = json.load(open(path, encoding="utf-8"))
    pairs = []

    for g in data:
        # format A: {"text": "...", "label": "..."}
        if isinstance(g, dict) and "label" in g and "text" in g:
            if not labels or g["label"] in labels:
                pairs.append((norm(g["text"]), g["label"]))
        # format B: {"text": "...", "entities": [[start, end, "LABEL"], ...]}
        elif isinstance(g, dict) and "text" in g and "entities" in g:
            txt = g["text"]
            for s, e, lbl in g["entities"]:
                if not labels or lbl in labels:
                    pairs.append((norm(txt[s:e]), lbl))
        # else: skip unknown rows

    return set(pairs)



def eval(data, gold_path="gold_entities.json", labels=None):
        
    try:

        
        import json, re
        if labels is None:
            labels = {"PERSON", "ORG", "GPE", "MONEY", "DATE"}  # keep it focused

        def norm(s):
            #This is a repair function that was was AI-generated
            s = s.lower()
            s = s.replace("’", "'").replace("“", '"').replace("”", '"').replace("–","-").replace("—","-")
            s = re.sub(r"(^[\"'“”‘’\(\[]+|[\"'“”‘’\)\],.:;!?]+$)", "", s.strip())  # strip edge punctuation
            s = re.sub(r"\s+", " ", s)
            return s

        # predictions from your table, filtered by label
        pred_set = {(norm(t), lab) for (t, lab, _) in data if lab in labels}

        # gold (text+label), filtered too
        gold = json.load(open(gold_path, encoding="utf-8"))
        #gold_set = {(norm(g["text"]), g["label"]) for g in gold if g["label"] in labels}
        gold_set = load_gold_pairs("gold_entities.json", labels={"PERSON","ORG","GPE","MONEY", "DATE"})

        tp = pred_set & gold_set
        fp = pred_set - gold_set         # extra / wrong label among the labels you care about
        fn = gold_set - pred_set         # missed among the labels you care about

        P = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
        R = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
        F1 = 2*P*R/(P+R) if (P+R) else 0.0

        print(f"\nPerformance on sample (labels={sorted(labels)}):")
        print(f"Precision={P:.3f}  Recall={R:.3f}  F1={F1:.3f}")
        print(f"Correct={len(tp)}  Wrong(extra)={len(fp)}  Missed={len(fn)}")

        if fp:
            print("\nFalse Positives:")
            for t,l in sorted(fp): print(f"  {t} ({l})")
        if fn:
            print("\nMissed (False Negatives):")
            for t,l in sorted(fn): print(f"  {t} ({l})")

        # Tell you what we ignored
        ignored_preds = [(t,l,_) for (t,l,_) in data if l not in labels]
        if ignored_preds:
            print(f"\nNote: Ignored {len(ignored_preds)} predictions with labels outside {sorted(labels)}.")
            input("\n\nPush any key to continue...")
    except Exception as e:
        error_msg(e)


def pie_setup(counts, title, min_pct=3):

    #I set up a pie chart here to visualize the data better.
    #A lot of this is boilerplate code I found that I augmented to fit
    #the assignment
    try:
        sns.set_theme()

        # Private helper functions
        def autopct_min(pct, min_pct=3):
            return f"{pct:.1f}%" if pct >= min_pct else ""

        def group_small_slices(counts, min_pct=3, other_label="Other"):
            try:
                total = counts.sum()
                small_mask = (counts / total * 100) < min_pct
                if small_mask.any():
                    small_sum = counts[small_mask].sum()
                    counts = counts[~small_mask]
                    counts[other_label] = small_sum
                return counts
            except Exception as e:
                error_msg(e)

        # --- prepare data BEFORE plotting ---
        s = group_small_slices(counts, min_pct=min_pct)

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = sns.color_palette("pastel", n_colors=len(s))

        wedges, texts, autotexts = ax.pie(
            s.values,
            labels=s.index,
            autopct=lambda pct: autopct_min(pct, min_pct=min_pct),
            startangle=90,
            colors=colors
        )
        ax.axis("equal")

        # Figure-level title (stays fixed) + a little top padding
        fig.suptitle(f"Entity Distribution by Label {title}", fontsize=14, y=0.98)

        # Nudge pie down a hair so it doesn't crowd the title
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.04, pos.width, pos.height])

        plt.show(block=False)
    except Exception as e:
        error_msg(e)



def eval_from_df(frame, gold_path="gold_entities.json", labels=None, single_word=False):
    import re
    if labels is None:
        labels = {"PERSON","ORG","GPE","FAC","MONEY","DATE"}  # tweak as you like

    def norm(s):
        #This part was AI-generated
        s = (s or "").lower()
        s = s.replace("’","'").replace("“",'"').replace("”",'"').replace("–","-").replace("—","-")
        s = re.sub(r"(^[\"'“”‘’\(\[]+|[\"'“”‘’\)\],.:;!?]+$)", "", s.strip())
        s = re.sub(r"\s+"," ", s)
        return s

    # predictions from the DF (Text, Entity)
    pred_set = {(norm(t), lab) for t, lab in zip(frame["Text"], frame["Entity"])
                if isinstance(t, str) and lab in labels}

    # gold pairs (supports both simple {"text","label"} and spaCy {"text","entities"} files)
    gold_set = load_gold_pairs(gold_path, labels=labels)

    # if we're evaluating single-word predictions, restrict gold to single-word too
    if single_word:
        gold_set = {(t,l) for (t,l) in gold_set if len(t.split()) == 1}

    tp = pred_set & gold_set
    fp = pred_set - gold_set
    fn = gold_set - pred_set

    P = len(tp) / (len(tp)+len(fp)) if (tp or fp) else 0.0
    R = len(tp) / (len(tp)+len(fn)) if (tp or fn) else 0.0
    F1 = 2*P*R/(P+R) if (P+R) else 0.0

    print(f"\nPerformance on sample ({'single-token only' if single_word else 'all entities'} | labels={sorted(labels)}):")
    print(f"Precision={P:.3f}  Recall={R:.3f}  F1={F1:.3f}")
    print(f"Correct={len(tp)}  Wrong(extra)={len(fp)}  Missed={len(fn)}")
    if fp:
        print("\nFalse Positives:")
        for t,l in sorted(fp): print(f"  {t} ({l})")
    if fn:
        print("\nMissed (False Negatives):")
        for t,l in sorted(fn): print(f"  {t} ({l})")



def data_analysis(data):
    #This is the function to compile the data into a data fram and count the entities
    #To avoid the fuction from become too convoluted, I made a separate function
    #to set up the graphs.
    try:
        print("\n\n")
        def multiword_filter(frame):
            try:
                # keep rows where Text has exactly 1 word
                mask = frame["Text"].astype(str).str.split().str.len() == 1
                return frame[mask].copy()
                
            except Exception as e:
                error_msg(e)
        
          # Build a pristine DF for evaluation
        df_raw = pd.DataFrame(data, columns=["Text", "Entity", "Description"])

        # ---- EVAL BEFORE ANY MUTATION ----
        eval_from_df(df_raw, gold_path="gold_entities.json", labels={"PERSON","ORG","GPE","FAC","MONEY","DATE"}, single_word=False)

        # Single-token eval (multiword removed)
        df_single_eval = df_raw[df_raw["Text"].astype(str).str.split().str.len() == 1].copy()
        eval_from_df(df_single_eval, gold_path="gold_entities.json", labels={"PERSON","ORG","GPE","FAC","MONEY","DATE"}, single_word=True)

      
        df = df_raw.copy()
        
        
                
        ent_total = df["Entity"].value_counts()
        small_labels = ent_total[ent_total < 20].index # 20 is the minimum bar graph threshold
        df["Entity"] = df["Entity"].apply(lambda x: "Other" if x in small_labels else x)

        #Recount of ent_total
        ent_total = df["Entity"].value_counts()        

                
        df_single = multiword_filter(df)          # The multiword filter is called from here
        ent_total_single = df_single["Entity"].value_counts()
        
        
        graph_setup(ent_total, "")
        graph_setup(ent_total_single, " — w/ Multiwords Removed")
        pie_setup(ent_total, "")
        pie_setup(ent_total_single, " — w/ Multiwords Removed") 
        plt.ioff() #This turns the interactive interface off

    except Exception as e:
        error_msg(e)




def main():
    try:

           
        model = spacy.load('en_core_web_sm')
        if not model:
            raise RuntimeError("No model detected!")
        
       
        
        model_setup(model)
        input("Press Enter to close the graphs...")
        
    except Exception as e:
        error_msg(e)


if __name__ == "__main__":
    main()