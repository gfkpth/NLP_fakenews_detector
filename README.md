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
- data
  - training_data.csv
  - testing_data.csv
- main.py                               # Preprocessing, model training, evaluation
  - helpers.py                          # Custom cleaning and vectorization functions
  - submission_logreg_tfidf.csv         # Example output file
- README.md

## Preprocessing Pipeline
- Lowercase conversion
- Removal of punctuation, digits, and special characters
- Stopword removal
- Cleaned version used for vectorization

## Feature Engineering
- TF-IDF: Vectorized using TfidfVectorizer(max_df=0.7)
- GloVe Embeddings: Word vectors averaged to form dense representation

## Models Trained
| Model               | Vector Type  | Notes                                                                  |
| ------------------- | ------------ | -----------------------------------------------------------------------|
| Logistic Regression | TF-IDF       | Highest accuracy, lightweight                                          |
| Random Forest       | TF-IDF       | Handles nonlinear relationships, performance slightly worse thanLogReg |
| K-Nearest Neighbors | TF-IDF       | Simpler baseline, performance somewhat worse                           |
| Logistic Regression | GloVe        | Dense embeddings, performs ok, but worse than TF-IDF                   |
| Random Forest       | GloVe        | More noise-sensitive with dense input                                  |
| XGBoost             | TF-IDF       | Strong performance (but see below), slower training                    |


## Hugging Face Models
- omykhailiv/bert-fake-news-recognition
- jy46604790/Fake-News-Bert-Detect
Used for inference on test headlines. Outputs mapped:
REAL → 1
FAKE → 0

Using these models directly without training leads to bad results (everything predicted to be fake).

## Evaluation
Performed on validation split (80/20) from training data using:
- Accuracy
- F1 Score
- Confusion Matrix
Transformers were not trained further but used directly for test-time inference.

# Predictions

Predictions were made for the models in various settings and saved in [results.csv](results.csv).

Also, the predictions for the unannotated test set was made based on our strongest model, a Logistic Regression, and written to [/data/testing_data_with_predictions.csv](testing_data_with_predictions.csv).

We also created predictions with the three other top-performing machine learning models and saved the results to [/data/testing_data_multiplepredictions_interannotatorcheck.csv](/data/testing_data_multiplepredictions_interannotatorcheck.csv).

For reasons currently unclear to us, the XGBClassifier performed very badly - it assigned 0 to all test items.
For a comparison of inter-annotator agreement we therefore excluded that model. Cohen's κ for the three remaining models (Logistic Regression, Random Forest, KNN) was 0.32, indicating 'fair agreement'.


# Experiment with transfer learning

The file [Transfer_experiment.py](Transfer_experiment.py) contains a brief proof of concept of using the Trainer class to actually train a pre-trained model on the current data. The results are in the range of an accuracy of > 0.98 after 2 epochs. In the interest of space, only the code is provided here.