# Iris Binary Logistic Regression

This project implements a binary logistic regression model on the Iris dataset using Python and scikit-learn. Although the Iris dataset originally has three classes, this project focuses on a binary classification task by selecting two classes from the dataset.

## Project Overview

- **Dataset:** Iris Dataset (filtered for binary classification)
- **Model:** Logistic Regression
- **Key Libraries:** scikit-learn, NumPy
- **Objective:** To predict the class of an Iris flower based on its features using a binary logistic regression model.

## Features

- **Data Loading:** Uses `load_iris` from scikit-learn.
- **Data Preprocessing:** Filters the dataset for binary classification and splits it into training and testing sets using `train_test_split`.
- **Model Building:** Implements a logistic regression model with `LogisticRegression`.
- **Evaluation:** Measures model performance with `accuracy_score`.

## Installation

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# For binary classification, select only the first two classes (0 and 1)
binary_filter = y < 2
X = X[binary_filter]
y = y[binary_filter]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

.Project Structure:

├── README       # This file
├── main.py      # The main script that loads data, builds the model, and evaluates performance.
└── requirements.txt  # Lists the required Python packages.

Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests with your improvements or additional features.

Acknowledgements
scikit-learn for providing the tools and datasets used in this project.
