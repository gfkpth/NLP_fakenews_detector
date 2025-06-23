# This file experiments with pre-trained transformer models

# %% Load libraries
import pandas as pd
from sklearn.model_selection import train_test_split

from nltk import word_tokenize, bigrams, trigrams
from nltk.metrics import agreement

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from wordcloud import WordCloud

import gensim.downloader as api # download pre-trained models

from transformers import pipeline
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import re
import string
import pickle


# auxiliary functions
import helper as helper
import importlib; importlib.reload(helper)



# %%
annotated = pd.read_csv('data/training_data.csv',sep='\t',header=None, names=["label", "text"])
annotated.head()


###############################
# Cleaning the dataset

# %% adding clean_text to annotated

# full cleaning
annotated['clean_text'] = annotated['text'].apply(helper.cleaning_strings)
# only remove extraneous spaces in 'text' field
annotated['text'] = annotated['text'].apply(helper.cleaning_strings,simpleedit=True)
# create a column without stop words
annotated['lemma'] = annotated['clean_text'].apply(helper.lemmatize)
annotated['lemma'] = annotated['lemma'].apply(helper.cleaning_strings)


#####################################
# Modelling
# %%
# Split the dataset into training and testing sets
X_train_tmp, X_test_tmp, y_train, y_test = train_test_split(annotated[['text','clean_text','lemma']], annotated['label'].values, test_size=0.2, random_state=42)

# %% creating X sets

X_train_clean = X_train_tmp['clean_text'].values
X_test_clean = X_test_tmp['clean_text'].values

X_train_lemma = X_train_tmp['lemma'].values
X_test_lemma = X_test_tmp['lemma'].values

X_train_dirty = X_train_tmp['text'].values
X_test_dirty = X_test_tmp['text'].values

# %%
X_train_clean

from datasets import Dataset

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": X_train_dirty, "label": y_train})
eval_dataset = Dataset.from_dict({"text": X_test_dirty, "label": y_test})




# load the relevant functions from HuggingFace and PyTorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Choose any classification model from the model hub
model_name = "omykhailiv/bert-fake-news-recognition"

# instantiate the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)



train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Remove the raw text to avoid redundancy
train_dataset = train_dataset.remove_columns(["text"])
eval_dataset = eval_dataset.remove_columns(["text"])

# Set format for PyTorch
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

from torch import nn
from transformers import Trainer
from transformers import TrainingArguments




training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %%

trainer.train()

# %%
trainer.evaluate()


# %%

# model prediction
output = model(input, output_hidden_states=False, output_attentions=False, return_dict=True)
probabilities = torch.softmax(output["logits"][0], -1).tolist()
label_names = model.config.id2label.values()
prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(probabilities, label_names)}
print(prediction)


# %% 

##########################################
# Transformer models
#
# beware when running these without GPU, will probably take a long time (even with GPU takes a little moment)

# %%
# Extract the headlines as lists (feed the full texts to the transformer models)
headlines_train = X_train_dirty.tolist()
headlines_test = X_test_dirty.tolist()

# %% run omykhailiv/bert-fake-news-recognition

helper.print_evaluation(None,
                        headlines_train,
                        headlines_test,
                        y_train,y_test,
                        'defaults',
                        model_id='omykhailiv-bert-fake-news-recognition',
                        vectype='default',
                        huggingpipe='omykhailiv/bert-fake-news-recognition')


# %% jy46604790/Fake-News-Bert-Detect

helper.print_evaluation(None,
                        headlines_train,
                        headlines_test,
                        y_train,y_test,
                        'defaults',
                        model_id='jy46604790-Fake-News-Bert-Detect',
                        vectype='default',
                        huggingpipe='jy46604790/Fake-News-Bert-Detect')


# Note how terribly both transformer models are performing here! Something must be off, for some reason they are actually predicting everything as fake


#######################
# Comparing results

# %%
results = pd.read_csv('results.csv').sort_values(by='accuracy',ascending=False)
results

# %%

print(results[['model_id','params','acc_train','accuracy']].to_latex(index=False, float_format="%.4f",escape=True))



##################################
# Predicting results for test.csv
# %%


# Load test data
test_df = pd.read_csv("data/testing_data.csv",sep='\t',header=None, names=["label", "text"])

# Clean the text
test_df['clean_text'] = test_df['text'].apply(helper.cleaning_strings)
test_df['clean_text'] = test_df['clean_text'].apply(helper.lemmatize)
test_df['clean_text'] = test_df['clean_text'].apply(helper.cleaning_strings)


# %%

# TF-IDF
X_test_final_tfidf = tfidf_vectorizer.transform(test_df['clean_text'])

# %%

# Predictions
test_df['logreg_tfidf'] = logreg_lemma.predict(X_test_final_tfidf)
test_df['rf_tfidf'] = rf_model_lemma.predict(X_test_final_tfidf)
test_df['knn_tfidf'] = knn_model_lemma.predict(X_test_final_tfidf)
test_df['xgb_tfidf'] = xgb_model_lemma.predict(X_test_final_tfidf)



# %%
test_df.head()

# %% to follow instructions, we use our logreg predictions for the final answer

test_df[['logreg_tfidf','text']].to_csv('data/testing_data_with_predictions.csv',sep='\t',header=False,index=False)


####################################
# Additional checks for curiosity
# %% but we also save a csv.file with our four predictions for later calculation of inter-annotator agreement
test_df.to_csv('data/testing_data_multiplepredictions_interannotatorcheck.csv',sep='\t',index=False)

# %% calculate inter-annotator agreement (using nltk.agreement)

# These are your "coders"
coders = ['logreg_tfidf', 'rf_tfidf', 'knn_tfidf', 'xgb_tfidf']

# %% it turns out that xgb_model_lemma is constantly predicting 0 like the transformers!
test_df['xgb_tfidf'].value_counts()

# %%
# Build the list of (coder, item, label) tuples
annotations = []

for idx, row in test_df.iterrows():
    for coder in coders:
        annotations.append((coder, idx, row[coder]))

ratingtask = agreement.AnnotationTask(data=annotations)
print("kappa for all four algorithms" +str(ratingtask.kappa()))
#print("fleiss " + str(ratingtask.multi_kappa()))
#print("alpha " +str(ratingtask.alpha()))
#print("scotts " + str(ratingtask.pi()))

# a kappa value of <.2 indicates no to only slight agreement for all four annotators

# %% just compare the three models that are actually working
coders = ['logreg_tfidf', 'rf_tfidf','knn_tfidf']

# Build the list of (coder, item, label) tuples
annotations = []

for idx, row in test_df.iterrows():
    for coder in coders:
        annotations.append((coder, idx, row[coder]))

ratingtask = agreement.AnnotationTask(data=annotations)
print("kappa for LogReg and randomforest: " +str(ratingtask.kappa()))

# kappa of 0.32 is 'fair agreement' for Cohen's Kappa

