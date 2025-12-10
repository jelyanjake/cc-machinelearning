import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# first things first, of course we need to load the dataset
df = pd.read_csv("complaints.csv")
df.head()

# then, we split the data
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# tf-idf vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# HERE WE TRAIN 3 MODELS: LOGISTIC REGRESSION, SVM, NAIVE BAYES
# so we can compare and see which one is best
# (requirement man gud ning sir carlo haha if i remember correctly)

# Logistic Regression
#logreg = LogisticRegression(max_iter=2000)
#logreg.fit(X_train_vec, y_train)
#logreg_pred = logreg.predict(X_test_vec)
#logreg_acc = accuracy_score(y_test, logreg_pred)

# SVM / Support Vector Machine
svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_pred = svm.predict(X_test_vec)
svm_acc = accuracy_score(y_test, svm_pred)

# Naive Bayes
#nb = MultinomialNB()
#nb.fit(X_train_vec, y_train)
#nb_pred = nb.predict(X_test_vec)
#nb_acc = accuracy_score(y_test, nb_pred)

# Print accuracies
#print("Logistic Regression:", logreg_acc)
#print("Linear SVM:", svm_acc)
#print("Naive Bayes:", nb_acc)

# we choose SVM as the best model based on accuracy

# And also print the classification report
print(classification_report(y_test, svm_pred))

# save model
import joblib
joblib.dump(svm, "complaint_classifier.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")