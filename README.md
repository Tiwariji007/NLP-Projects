# NLP-Projects
# Project - 1
# Spam Mail Detection using Naive Bayes

This project demonstrates how to build a spam Mail detection model using the Naive Bayes algorithm in Python. The model is trained on a dataset containing Email messages labeled as spam or ham (non-spam).

## Dataset

The dataset used in this project is available under the directory `/kaggle/input/sms-spam-collection-dataset/spam.csv`. It contains SMS messages labeled as 'spam' or 'ham'. The dataset has the following columns:

- `v1`: Label indicating whether the SMS is spam or ham.
- `v2`: Text content of the SMS.
- `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`: Additional columns with NaN values, not used in this project.

## Implementation

The project is implemented in Python using the following libraries:

- `numpy` and `pandas` for data manipulation and analysis.
- `sklearn` for building and evaluating machine learning models.

The main steps of the implementation include:

1. Data loading: The dataset is loaded into a pandas DataFrame.
2. Data preprocessing: The label column `v1` is transformed into binary labels (0 for ham, 1 for spam).
3. Model training: The Naive Bayes model is trained using the CountVectorizer for feature extraction.
4. Model evaluation: The trained model is evaluated on a test set using classification metrics such as precision, recall, and F1-score.

## Usage

You can run the provided Python code in a Python 3 environment. If you're using Jupyter Notebook or Google Colab, simply copy and paste the code into a new notebook cell and execute it. Make sure to have the required libraries installed.

## Quick Method

A quick method using scikit-learn's `Pipeline` is also demonstrated in the provided code. It combines the feature extraction and model training steps into a single pipeline for convenience.


# Project - 2

# Kaggle News Category Dataset Analysis

## Introduction
In this analysis, we explore the Kaggle News Category Dataset and build a classifier to predict the category of news articles based on their short descriptions. We preprocess the text data using spaCy and then train a Multinomial Naive Bayes classifier using CountVectorizer as a feature extraction technique.

## Environment Setup
This analysis is performed in a Python 3 environment using the kaggle/python Docker image. Necessary libraries such as NumPy, pandas, and scikit-learn are installed.

## Dataset
The dataset used in this analysis is the Kaggle News Category Dataset. It contains news articles categorized into various topics such as politics, entertainment, sports, etc.

## Code Overview
The analysis includes the following steps:

1. **Loading the Dataset**: The dataset is loaded from the provided JSON file using pandas.

2. **Data Preprocessing**: We balance the dataset by sampling an equal number of articles from each category. We then preprocess the short descriptions of the articles using spaCy for tokenization, lemmatization, and removing stopwords and punctuation.

3. **Model Training**: We split the preprocessed data into training and testing sets. We train a Multinomial Naive Bayes classifier using CountVectorizer to convert the text data into numerical feature vectors.

4. **Model Evaluation**: We evaluate the performance of the trained classifier on the test data using classification metrics such as precision, recall, and F1-score.

## Results
The classifier achieves an overall accuracy of around 26% on the test data. The classification report provides detailed metrics for each category, including precision, recall, and F1-score.

## Conclusion
Despite the relatively low accuracy, the classifier demonstrates some capability in predicting the category of news articles based on their short descriptions. Further improvements could be made by experimenting with different preprocessing techniques, feature extraction methods, and machine learning algorithms.

## Author
These project is created by "Adarsh Tiwari". You can reach out to me at "Tiwariadarshformal@gmail.com" for any inquiries or feedback.

