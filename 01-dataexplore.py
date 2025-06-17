# Preamble

# overall plan
# 
# 1) Data exploration
# 2) Prepare cleaning function
# 3) vectorisation
#   - BOW
#   - TF-IDF
#   - embedding (word2vec etc-)
# 4) training 
#   - ML models
#   - transformers
# 5) validation
# 6) test (on labelled data)
# 7) make predictions for "testing_data.csv"

# auxiliary functions
#
# - data_cleaning(string), return string
#
# - training_validation(model,train_df,val_df,otheroptions)
#    - possibly return history (or directly write log)
#
# - report_fn(model,test_df): logs results to csv
#   - log: timestamp, modelname, train_acc, test_acc, test_precision, test_recall, test_f1, parameters


# deliverables
# - documented code (possibly also README.md in git repo)
# - presentation
# - predictions for testing_data.csv

# %%
import pandas as pd
from sklearn.model_selection import train_test_split






# Data exploration

# Plan: 
# 1. explore dataset
# 2. identify necessary steps of pre-processing
# 3. combine into one function for data cleaning to export to auxiliary.py

# %%
annotated = pd.read_csv('data/training_data.csv',sep='\t',header=None, names=["label", "text"])
annotated




# %% 
# Basic data overview

annotated['text'].head()
print('Shape of the dataset:', annotated.shape)
print('Label counts:', annotated['label'].value_counts())

# %%

print('Test for missing values:', annotated.isnull().sum())



# %%
