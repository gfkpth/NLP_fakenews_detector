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

# %% Load libraries
import pandas as pd
from sklearn.model_selection import train_test_split

from nltk import word_tokenize, bigrams, trigrams

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

# auxiliary functions
import helper as helper
import importlib; importlib.reload(helper)



#####################
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
plt.figure(figsize=(8, 5))
sns.histplot(annotated['text_length'], bins=30, kde=True)
plt.title("Distribution of Headline Lengths")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.show()
# %%
# World cloud for real and fake news


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
plt.savefig('assets/wordcloud-real.png')
plt.show()
# %%
# Plot word cloud for fake news
plt.figure(figsize=(12, 6))
wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(' '.join(fake_news))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Fake News")
plt.savefig('assets/wordcloud-fake.png')
plt.show()


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

###############################
# Cleaning the dataset

# %% adding clean_text to annotated

# full cleaning
annotated['clean_text'] = annotated['text'].apply(helper.cleaning_strings)
# only remove extraneous spaces in 'text' field
annotated['text'] = annotated['text'].apply(helper.cleaning_strings,simpleedit=True)
# create a column without stop words
annotated['no_stop'] = annotated['clean_text'].apply(helper.remove_stop)


#####################################
# Modelling
# %%
# Split the dataset into training and testing sets
X_train_tmp, X_test_tmp, y_train, y_test = train_test_split(annotated[['text','clean_text','no_stop']], annotated['label'].values, test_size=0.2, random_state=42)

# %% creating X sets

X_train = X_train_tmp['no_stop'].values
X_test = X_test_tmp['no_stop'].values

X_train_clean = X_train_tmp['clean_text'].values
X_test_clean = X_test_tmp['clean_text'].values

X_train_dirty = X_train_tmp['text'].values
X_test_dirty = X_test_tmp['text'].values


#############
# Vectorisation
# %% Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,  # Limit to 1000 features
    stop_words='english',  # Remove English stop words
    ngram_range=(1, 2),  # Use unigrams and bigrams
    token_pattern=r'\b\w+\b'  # Tokenize words
)
# Fit and transform on cleaned text data
X_train_vectf = tfidf_vectorizer.fit_transform(X_train_clean).toarray()
X_test_vectf = tfidf_vectorizer.transform(X_test_clean).toarray()

X_train_vectf.shape



# %% vectorising with GloVE
glove = api.load("glove-wiki-gigaword-100")
glove_size=100
X_train_glove = helper.dense_vectorize_text(X_train,glove,vector_size=glove_size)
X_test_glove = helper.dense_vectorize_text(X_test,glove,vector_size=glove_size)



#################################
# Model training and evaluation


# %% Logistic regression
# Create a logistic regression model
max_iter=1000
logreg = LogisticRegression(max_iter=max_iter,random_state=5,n_jobs=-1)
# Train the model
logreg.fit(X_train_vectf, y_train)

# Evaluate
helper.print_evaluation(logreg, X_train_vectf, X_test_vectf, y_train, y_test,f'max_iter={max_iter}',model_id=f'logreg_{max_iter}',vectype='tf-idf')


# %% Logistic regression with GloVe
max_iter=5000
logreg_glove = LogisticRegression(max_iter=max_iter,random_state=5,n_jobs=-1)
# Train the model
logreg_glove.fit(X_train_glove, y_train)
# Evaluate
helper.print_evaluation(logreg_glove, X_train_glove, X_test_glove, y_train, y_test,f'max_iter={max_iter}',model_id=f'logreg_glove_{max_iter}',vectype=f'glove_{glove_size}')



# %% Create a Random Forest model
nestim=100
# Create a random forest classifier
rf_model = RandomForestClassifier(n_estimators=nestim, random_state=42,n_jobs=-1)
# Train the model
rf_model.fit(X_train_vectf, y_train)

# Evaluate
helper.print_evaluation(rf_model, X_train_vectf, X_test_vectf, y_train, y_test,f'n_estimators={nestim}',model_id='rndforest_1',vectype='tf-idf')

# %% another random forest
nestim=1000
max_depth=None
min_samp_leaf=2
# Create a random forest classifier
rf_model = RandomForestClassifier(n_estimators=nestim, min_samples_leaf=min_samp_leaf,max_depth=max_depth, random_state=42,n_jobs=-1)
# Train the model
rf_model.fit(X_train_vectf, y_train)

# Evaluate
helper.print_evaluation(rf_model, X_train_vectf, X_test_vectf, y_train, y_test,f'n_estimators={nestim},max_depth={max_depth},min_samp_leaf={min_samp_leaf}',model_id=f'rndforest_{nestim}',vectype='tf-idf')

# %% Random Forest with GloVe
nestim=100
rf_glove = RandomForestClassifier(n_estimators=nestim, random_state=42,n_jobs=-1)
# Train the model
rf_glove.fit(X_train_glove, y_train)
# Evaluate
helper.print_evaluation(rf_glove, X_train_glove, X_test_glove, y_train, y_test,f'n_estimators={nestim}',model_id='rndforest_1',vectype=f'glove_{glove_size}')



# %%
# Create KNN model
neigh= 3
# Create a KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=neigh)
# Train the model
knn_model.fit(X_train_vectf, y_train)

# Evaluate
helper.print_evaluation(knn_model, X_train_vectf, X_test_vectf, y_train, y_test,f'k={neigh}',model_id='knn_3',vectype='tf-idf')



# %% XGBoost

xgb_model = xgb.XGBClassifier(random_state=1)
xgb_model.fit(X_train_vectf, y_train)

# Evaluate
helper.print_evaluation(xgb_model, X_train_vectf, X_test_vectf, y_train, y_test,f'defaults',model_id='xgb_1',vectype='tf-idf')

# %% XGBoost tweaking
nestim=500
max_depth=100
lr=0.3
xgb_model = xgb.XGBClassifier(n_estimators=nestim,max_depth=max_depth, learning_rate=lr, random_state=1)
xgb_model.fit(X_train_vectf, y_train)

# Evaluate
helper.print_evaluation(xgb_model, 
                        X_train_vectf, 
                        X_test_vectf, 
                        y_train, 
                        y_test,
                        f'n_estim={nestim},max_depth={max_depth},lr={lr}',
                        model_id=f'xgb_{nestim}_{max_depth}_{lr}',
                        vectype='tf-idf')



##########################################
# Transformer models
#
# beware when running these without GPU, will probably take a long time (even with GPU takes a little moment)

# %%
# Extract the headlines as lists (feed the full texts to the transformer models)
headlines_train = X_train_dirty.tolist()
headlines_test = X_test_dirty.tolist()

# %%
headlines_train[:10]


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

# %% result overview
results =pd.read_csv('results.csv').sort_values(by='accuracy',ascending=False)
results


