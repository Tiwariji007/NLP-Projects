# NLP-Projects
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

## Author

This project is created by "Adarsh Tiwari". You can reach out to me at "Tiwariadarshformal@gmail.com" for any inquiries or feedback.

