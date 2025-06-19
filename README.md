# NLP_fakenews_detector using NLP

## Overview
This project tackles the problem of identifying fake news headlines using Natural Language Processing (NLP). We used a labeled dataset to train multiple pmodels, including Logistic Regression, Random Forest, XGB, and KNN, with both TF-IDF and GloVe text representations. Additionally, we integrated pre-trained transformer models from Hugging face to compare performance. The final goal was to accurately classify unseen headlines as real or fake and generate prediction files in the required format. Throughout the process, we applied data cleaning, feature engineering, model evaluation and inference techniques to build a robust fake news detecction pipeline.
 
## Objectives
- Clean and preprocess real-world text data
- Apply TF-IDF and GloVe-based feature engineering
- Train and evaluate multiple machine learning models
- Integrate Hugging Face transformer models for inference
- Generate prediction files on unseen test data
- Compare classical ML vs transformer model performance

## Dataset
### Training Data: 'training_data.csv'
- text: The news headline
- label: 0 = fake, 1 = real

### Testing Data: 'testing_data.csv'
- text: News headline only (unlabeled)

### Output: 
- Prediction file with the same format as test file, but `label` column filled with predicted values

## Project Structure


## Preprocessing Pipeline
- Lowercase conversion
- Removal of punctuation, digits, and special characters
- Stopword removal
- Cleaned version used for vectorization

## Feature Engineering
- TF-IDF: Vectorized using TfidfVectorizer(max_df=0.7)
- GloVe Embeddings: Word vectors averaged to form dense representation

## Models Trained
| Model               | Vector Type  | Notes                                  |
| ------------------- | ------------ | -------------------------------------- |
| Logistic Regression | TF-IDF       | High accuracy, lightweight             |
| Random Forest       | TF-IDF       | Handles nonlinear relationships        |
| K-Nearest Neighbors | TF-IDF       | Simpler baseline, slower on large data |
| Logistic Regression | GloVe        | Dense embeddings, performs well        |
| Random Forest       | GloVe        | More noise-sensitive with dense input  |
| XGBoost             | TF-IDF       | Strong performance, faster than RF     |


## Hugging Face Models
- omykhailiv/bert-fake-news-recognition
- jy46604790/Fake-News-Bert-Detect
Used for inference on test headlines. Outputs mapped:
REAL → 1
FAKE → 0

## Evaluation
Performed on validation split (80/20) from training data using:
- Accuracy
- F1 Score
- Confusion Matrix
Transformers were not trained further but used directly for test-time inference.

# Predictions