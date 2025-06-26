# classification.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from numpy import genfromtxt, zeros
import os

# Load dataset
data = genfromtxt('data/iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
labels = genfromtxt('data/iris.csv', delimiter=',', usecols=(4), dtype=str)

# Print data shape
print("Data shape:", data.shape)
print("Label shape:", labels.shape)

# Convert class labels to numeric
t = zeros(len(labels))
t[labels == 'setosa'] = 0
t[labels == 'versicolor'] = 1
t[labels == 'virginica'] = 2

# Visualize: Sepal length vs Petal length
plt.figure()
plt.scatter(data[t == 0, 0], data[t == 0, 2], c='blue', label='Setosa')
plt.scatter(data[t == 1, 0], data[t == 1, 2], c='red', label='Versicolor')
plt.scatter(data[t == 2, 0], data[t == 2, 2], c='green', label='Virginica')
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.title("Iris Classification - Feature Scatter")
plt.legend()
plt.tight_layout()
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, t, test_size=0.4, random_state=0)

# Model: Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# Classification report and confusion matrix
predictions = model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions,
                                                          target_names=['Setosa', 'Versicolor', 'Virginica']))

# Cross-validation score
cv_scores = cross_val_score(model, data, t, cv=6)
print("Cross-validation scores:", cv_scores)
print("Mean CV score: {:.2f}".format(np.mean(cv_scores)))


