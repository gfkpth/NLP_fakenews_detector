# auxiliary functions

# import
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk import word_tokenize, bigrams, trigrams, pos_tag
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,classification_report, confusion_matrix


from transformers import pipeline
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import re
import string
import os
import csv

# string cleaning
# currently not removing numbers, consider adding?
def cleaning_strings(iostring,lower=True,simpleedit=False):
    # patterns
    multispace = re.compile(r'\s+')
    wrongposs=re.compile(r'‚s')     # this is a strange type of possessive apostrophe
    
    if simpleedit:
        iostring = multispace.sub(' ', iostring)
        iostring = wrongposs.sub(r"'s",iostring)    # for simpleedit replace apostrophes with more common form
    else:
    
        # check for missing spaces in front of capitalised words or after punctuation
        # and insert space
        missingspace = re.compile(r'([\.]|(?:[A-Za-z]*[a-z]))([A-Z][A-Za-z]*)')
        iostring = missingspace.sub(r'\1 \2',iostring)
        
        # set up patterns for deletion
        realhyphen = re.compile(r'[^A-Za-z]+-[^A-Za-z]*')
        punctnohyphen = ''.join(list(filter(lambda x: x if x != '-' else '',string.punctuation))) + ','
        
        
        # delete patterns
        
        specialsigns = re.compile(f"[{re.escape(punctnohyphen)}]")
        
        # processing the patterns
        spacepattern = [realhyphen,multispace]
        delpattern = [wrongposs,specialsigns]
        for p in spacepattern:
            iostring = p.sub(' ', iostring)

        for p in delpattern:
            iostring = p.sub('', iostring)
        
    if lower:
        return iostring.lower()
    else:
        return iostring


def remove_stop(inpstring):
    stop_words = set(stopwords.words('english'))    
    return ' '.join(list(filter(lambda x: '' if x in stop_words else x, word_tokenize(inpstring))))

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0]#  Get POS tag's first character (e.g., 'N' from 'NN')
    #Maps it to a WordNet-compatible tag
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
    return tag_dict.get(tag, wordnet.NOUN) # returns the word type (Noun if we have not found)
    
def lemmatize(lst):
    wordnet_lemma  = WordNetLemmatizer()
    return [wordnet_lemma.lemmatize(word,pos=get_wordnet_pos(word)) for word in lst]

    
    
def dense_vectorize_text(texts, model, vector_size=100):
    vectors = []
    for text in texts:
        word_vectors = []
        for word in text.split():  # assuming `text` is a string of words
            if word in model:
                word_vectors.append(model[word])
        if word_vectors:
            vectors.append(np.mean(word_vectors, axis=0))  # mean pooling
        else:
            vectors.append(np.zeros(vector_size))  # fallback for unknown words
    return np.array(vectors)    
    

# Function to run predictions with a Hugging Face pipeline
def run_huggingface_pipeline(model_name, data):
    print(f"\nRunning predictions using model: {model_name}")
    classifier = pipeline(task="text-classification", model=model_name, top_k=1)
    predictions = []

    for output in tqdm(classifier(data, truncation=True)):
        label = output[0]["label"]
        predictions.append(1 if label.lower() == "real" else 0)

    return predictions


# print evaluation and append results to results 
def print_evaluation(model,X_train,X_test,y_train,y_test,params,model_id=None,vectype='tf-idf',huggingpipe=False,printout=True,csvout='results.csv',saveconfusion=True):
    """ Print model evaluation
    
    Arguments:
    - **model**: the model for evaluation

    - **model_id**: a string identifier for the model for logging
    - **csvout**: filepath for csv file logging the results, set to '' to disable logging
    - **printout**: set to False if no printed output is desired
     
    return:
    - classification report for storage and further comparison
    """
    cycle = ['train', 'test']
    resdict = {
        'train': {
            'X': X_train,
            'y_obs': y_train
        },
        'test': {
            'X': X_test,
            'y_obs': y_test
        }
    }
    
    
    for c in cycle:
        # Predict class probabilities
        if not huggingpipe:
            resdict[c]['y_pred'] = model.predict(resdict[c]['X'])
        else:
            resdict[c]['y_pred'] = run_huggingface_pipeline(huggingpipe, resdict[c]['X'])
    
        # Compute metrics
        resdict[c]['acc'] = accuracy_score(resdict[c]['y_obs'], resdict[c]['y_pred'])
        resdict[c]['prec'] = precision_score(resdict[c]['y_obs'], resdict[c]['y_pred'], average='weighted')
        resdict[c]['rec'] = recall_score(resdict[c]['y_obs'], resdict[c]['y_pred'], average='weighted')
        resdict[c]['f1'] = f1_score(resdict[c]['y_obs'], resdict[c]['y_pred'], average='weighted')


        report = classification_report(resdict[c]['y_obs'], resdict[c]['y_pred'])

        if printout:
            print(f'Results for {c} set')
            print()
            print(f"Accuracy: {resdict[c]['acc']:.4f}")
            print(f"Precision: {resdict[c]['prec']:.4f}")
            print(f"Recall: {resdict[c]['rec']:.4f}")
            print(f"F1 Score: {resdict[c]['f1']:.4f}")
            print(report)
        
        
            # Predict & confusion matrix
            cm = confusion_matrix(resdict[c]['y_obs'], resdict[c]['y_pred'])

            plt.figure(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for {c} set')
            
            if saveconfusion:
                plt.savefig('assets/' + model_id + '_' + c + '.png')
            plt.show()
    
    if csvout:
        write_results(model_id,vectype,resdict,params,file=csvout)
    
    #return resdict


# function to append results to csv
def write_results(modid,vectype,resdict,params,file='results.csv'):

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fields=[timestamp, 
            modid, 
            vectype,
            resdict['train']['acc'], 
            resdict['train']['f1'], 
            resdict['test']['acc'], 
            resdict['test']['prec'], 
            resdict['test']['rec'], 
            resdict['test']['f1'], 
            params]
    headers = ['time','model_id', 'vec_type', 'acc_train', 'f1_train', 'accuracy', 'precision', 'recall', 'f1_score','params']
    
    
    file_exists = os.path.exists(file)

    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(fields)
        
