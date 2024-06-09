# Financial Document Classification using Machine Learning

This project aims to classify financial documents into categories such as Balance Sheets, Cash Flow, Income Statements, Notes, and Others using machine learning techniques.

## Introduction

Text classification is crucial in Natural Language Processing (NLP), where the goal is to categorize text data into predefined labels. In this project, we focus on classifying financial documents based on their textual content.

## Features

-Extracts text from HTML files.

-Cleans and preprocesses the extracted text.

-Transforms the text into TF-IDF vectors.

-Predicts the category of the financial statement using a pre-trained classification model.

## Requirements

-Python 3.1

-Streamlit

-NLTK

-BeautifulSoup4

-scikit-learn

-pickle

## Approach

### Data Sources

The dataset comprises HTML files containing various financial documents, each labeled with one of the five categories.

### Text Extraction and Cleaning

We extracted text from HTML files using BeautifulSoup and cleaned it by removing HTML tags, special characters, and stopwords.

### Feature Extraction

Text features were extracted using the TF-IDF vectorizer, converting text data into numerical features that capture the importance of words across documents.

### Model Comparison

We trained and evaluated multiple machine learning models, including Decision Tree, Random Forest, Multinomial Naive Bayes, K-Nearest Neighbors, and Support Vector Classifier.

### Model Training and Evaluation

The models were trained on a labeled dataset and evaluated using standard metrics such as accuracy, precision, recall, and F1-score.

### Model Selection

Based on performance metrics, the Support Vector Classifier (SVC) with a linear kernel was selected as the best-performing model.

### Hyperparameter Tuning

Hyperparameter tuning using grid search was performed to optimize the SVC model's performance.

### Prediction

The trained SVC model and TF-IDF vectorizer were used to predict the category of new financial documents.

## Setup Instructions

**Step 1**: Install Required Libraries

Install the required libraries using pip:

pip install streamlit nltk beautifulsoup4 scikit-learn

**Step 2**: Download NLTK Data

Download the necessary NLTK data files:

nltk.download('stopwords')

nltk.download('punkt')

**Step 3**: Prepare the Models

Place pre-trained models (tfidf_vectorizer_1.pkl and Findoc_Classification_model.pkl) in the specified directory.

**Step 4**: Run the Application

Save the main script as findoc_classification.py and run it using Streamlit:

streamlit run findoc_classification.py

## Conclusion

This project successfully developed a machine learning model for financial document classification. Future work may involve exploring advanced NLP techniques and expanding the dataset for further improvements.

## Acknowledgements

-Streamlit

-NLTK

-BeautifulSoup

-scikit-learn
