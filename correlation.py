# Correlation Heatmap Script for Iris Dataset

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Compute correlation matrix (excluding target)
corr_matrix = df.iloc[:, :-1].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, square=True)
plt.title('Correlation Heatmap of Iris Features')
plt.xticks(rotation=20)
plt.yticks(rotation=20)
plt.tight_layout()
plt.show()
