<p align="left"> <!-- Python version --> <img src="https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white" /> <!-- Build status (dummy until you connect CI) --> <img src="https://img.shields.io/badge/Build-Passing-brightgreen?logo=githubactions&logoColor=white" /> <!-- License --> <img src="https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white" /> <!-- Stars --> <img src="https://img.shields.io/github/stars/yourusername/yourrepo?style=social" /> <!-- Issues --> <img src="https://img.shields.io/github/issues/yourusername/yourrepo" /> <!-- PRs --> <img src="https://img.shields.io/github/issues-pr/yourusername/yourrepo?color=blue" /> </p>

Complaint Classification System:
A machine learning pipeline for automatically classifying consumer complaints into predefined categories using text classification techniques.

📋 Project Overview
This project implements a text classification system that can categorize consumer complaints (e.g., billing issues, service problems, product defects) using natural language processing and machine learning algorithms. The system compares multiple classification models to identify the best performer for the task.

🚀 Features
Text Preprocessing: Automatic removal of stop words and TF-IDF vectorization

Multiple Model Comparison: Tests three different classification algorithms

Model Evaluation: Comprehensive performance metrics including accuracy, precision, recall, and F1-score

Model Persistence: Saves trained models for deployment

Reproducible Results: Fixed random seed for consistent performance

🏗️ Model Architecture
The system implements and compares three classification algorithms:

Linear Support Vector Machine (SVM) - Best performing model

Multinomial Naive Bayes - Probabilistic classifier

Logistic Regression - Currently commented out in the code

📊 Dataset
The model expects a CSV file named complaints.csv with two columns:

text: The complaint description (string)

label: The complaint category (string or integer)

📈 Performance Metrics
The model evaluation includes:

Accuracy: Overall correct predictions

Precision: Correct positive predictions among predicted positives

Recall: Correct positive predictions among actual positives

F1-Score: Harmonic mean of precision and recall

Confusion Matrix: Visual representation of classification performance

🧪 Model Selection
Based on typical performance characteristics:

SVM: Often best for text classification with high-dimensional features

Naive Bayes: Fast training, works well with text data

Logistic Regression: Good baseline, interpretable coefficients

The current implementation selects SVM as the primary model based on accuracy.

🔄 Workflow
Data Preparation: Load and split complaint data

Feature Extraction: Convert text to TF-IDF vectors

Model Training: Train multiple classifiers

Evaluation: Compare model performance

Selection: Choose best performing model

Persistence: Save model for deployment

📝 Requirements
Python 3.7+

pandas >= 1.0

numpy >= 1.18

scikit-learn >= 0.24

seaborn >= 0.11

matplotlib >= 3.3

joblib >= 1.0
