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
import helper as helper
import importlib; importlib.reload(helper)




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
!pip install wordcloud
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
# test cleaning function
teststring = 'this is a pro-trump rallying 4534-324 inLondon %34 f cry'
print(helper.cleaning_strings(teststring))

#

# %% adding clean_text to annotated

annotated['clean_text'] = annotated['text'].apply(helper.cleaning_strings)




# %%


# bi_tokens = list(bigrams(tokens))
# print(tokens)
# print(bi_tokens)

# %%
# Text Vectorization(TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,  # Limit to 1000 features
    stop_words='english',  # Remove English stop words
    ngram_range=(1, 2),  # Use unigrams and bigrams
    token_pattern=r'\b\w+\b'  # Tokenize words
)
# Fit and transform on cleaned text data
X = tfidf_vectorizer.fit_transform(annotated['clean_text']).toarray()
y = annotated['label']

# %%
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#################################
# Model training and evaluation


# %%
# Logistic regression
from sklearn.linear_model import LogisticRegression
# Create a logistic regression model
logreg = LogisticRegression(max_iter=1000,random_state=5)
# Train the model
logreg.fit(X_train, y_train)

# Evaluate
helper.print_evaluation(logreg, X_train, X_test, y_train, y_test,'max_iter=1000',model_id='logreg_1000',vectype='tf-idf')


# %%
# Create a Random Forest model
from sklearn.ensemble import RandomForestClassifier
nestim=100
# Create a random forest classifier
rf_model = RandomForestClassifier(n_estimators=nestim, random_state=42)
# Train the model
rf_model.fit(X_train, y_train)

# Evaluate
helper.print_evaluation(rf_model, X_train, X_test, y_train, y_test,f'n_estimators={nestim}',model_id='rndforest_1',vectype='tf-idf')

# %%
# Create KNN model
from sklearn.neighbors import KNeighborsClassifier
neigh= 5
# Create a KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=neigh)
# Train the model
knn_model.fit(X_train, y_train)

# Evaluate
helper.print_evaluation(knn_model, X_train, X_test, y_train, y_test,f'k={neigh}',model_id='knn_5',vectype='tf-idf')

# %%
