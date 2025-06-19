---
output:
  beamer_presentation:
    pdf-engine: xelatex
    listings: true
title: Fake news detection
author: [Simbiat Musa, Georg F.K. Höhn]
short-author: [Musa, Höhn]
institute: Ironhack
date: 19 June 2025
section-titles: true
tables: true
indent: true
theme: default
colortheme: greenmeadow
bibliography: /home/georg/academia/BibTeXreferences/literature.bib
babel-lang: english
lang: en-GB
language: english
mainfont: Linux Biolinum
monofont: DejaVu Sans Mono
fontsize: 13pt
papersize: a4paper
numbersections: true
csquotes: true
---

## Overview
This project tackles the problem of identifying fake news headlines using Natural Language Processing(NLP). A labeled dataset was used to tran multiple models including Logistic Regression, Random Forest, KNN and XGB with both TF-IDF and GloVe text representation. 
\tableofcontents

# Introduction
This project focuses on detecting fake news headlines by applying a full end-end machine learning pipeline. Using a labeled dataset of news headlines, we trained several classical machine learning models and also leveraged state of the art transformer models from hugging face. We evaluated and compared model performance and ultimately produced prediction files to classify unseen news headlines as real or fake.
- 34152 rows of headlines annoted as `fake news` (0) or `real news` (1)



### Aims

Classify texts in a test set as `fake` or `real`


# Data overview

##

- fake news: 17572
- real news: 16580
- relatively balanced


# Methodology

## 




# Brief word about transformer models

## 

- tried pipelines with 
  1) `jy46604790/Fake-News-Bert-Detect`
  2) `omykhailiv/bert-fake-news-recognition`
- both performed abysmally: everything is fake, see result for 1) below
- not sure why, over-sensitive? something wrong with our data pre-processing?

![](../assets/jy46604790-Fake-News-Bert-Detect_test.png){height="50%}


# Training results

## 




# Conclusion

##


##

### Collaboration

- used py files to avoid notebook consistency issues with git
- VS Code offers the possibility of generating jupyter-like cells with `# %%` 


## 

\centering\LARGE Thanks for your attention!


