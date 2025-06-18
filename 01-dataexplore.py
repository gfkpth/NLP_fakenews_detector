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

from nltk import word_tokenize, bigrams, trigrams

import re
import string

# auxiliary functions
from auxiliary import *



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

print(annotated['text'].head())

print('Shape of the dataset:', annotated.shape)
print('Label counts:', annotated['label'].value_counts())


# %%

print('Test for missing values:', annotated.isnull().sum())

# none, very good

# %%
# Add a column for text length
annotated['text_length'] = annotated['text'].apply(len)


# Describe text lengths
print(annotated['text_length'].describe())

# Plot distribution
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(annotated['text_length'], bins=30, kde=True)
plt.title("Distribution of Headline Lengths")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.show()
# %%
# World cloud for real and fake news
from wordcloud import WordCloud

# Seperate headlines by label
real_news = annotated.loc[annotated['label'] == 1,'text']
fake_news = annotated.loc[annotated['label'] == 0,'text']


# %%

# Plot word cloud for real news
plt.figure(figsize=(12, 6))
wordcloud_real = WordCloud(width=800, height=400, background_color='white').generate(' '.join(real_news))
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Real News")
plt.show()
# %%
# Plot word cloud for fake news
plt.figure(figsize=(12, 6))
wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(' '.join(fake_news))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Fake News")
plt.show()
# %%

# Let's search for special characters we need to take care of

# %%
annotated.loc[annotated.text.str.contains(r'\*|%|\)|\(')]


# %%

# find hyphens that ar not connected to letters on both sides (that are connected to at least one non-letter on the left)
realhyphen = re.compile(r'[^A-Za-z]+-[^A-Za-z]*')

testphrase = "this test-(45)-bi is a very-important (but not so)-important test - and we want to get rid of free-standing hyphens, too"

realhyphen.sub(' ',testphrase)

# %% The tokeniser actually does a good job of dealing with hyphens as well
# 

tokens = word_tokenize(testphrase)
print(tokens)


# %%

punctnohyphen = list(filter(lambda x: x if x != '-' else '',string.punctuation))
punctnohyphen

#re.escape(string.punctuation)



# %%


# bi_tokens = list(bigrams(tokens))
# print(tokens)
# print(bi_tokens)

# %%
