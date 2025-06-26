# Iris Clustering Script (using KMeans)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import completeness_score, homogeneity_score

# Load the Iris dataset
data = load_iris()
X = data.data
y_true = data.target
feature_names = data.feature_names

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, init='random', random_state=42)
kmeans.fit(X)
y_pred = kmeans.labels_

# Evaluate clustering performance
completeness = completeness_score(y_true, y_pred)
homogeneity = homogeneity_score(y_true, y_pred)

print("Completeness Score:", completeness)
print("Homogeneity Score:", homogeneity)

# Plot real classes
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[y_true == 0, 0], X[y_true == 0, 2], c='blue', label='Setosa')
plt.scatter(X[y_true == 1, 0], X[y_true == 1, 2], c='red', label='Versicolor')
plt.scatter(X[y_true == 2, 0], X[y_true == 2, 2], c='green', label='Virginica')
plt.title('True Labels')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[2])
plt.legend()

# Plot predicted clusters
plt.subplot(1, 2, 2)
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 2], c='purple', label='Cluster 0')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 2], c='orange', label='Cluster 1')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 2], c='cyan', label='Cluster 2')
plt.title('KMeans Clusters')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[2])
plt.legend()

plt.tight_layout()
plt.show()
