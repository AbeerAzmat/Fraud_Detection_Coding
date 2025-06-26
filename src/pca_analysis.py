# Iris PCA (Dimensionality Reduction) Script

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame for easy plotting
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['species'] = pd.Categorical.from_codes(y, target_names)

# Plot the PCA results
colors = ['blue', 'red', 'green']
plt.figure(figsize=(8,6))

for color, target in zip(colors, target_names):
    subset = df_pca[df_pca['species'] == target]
    plt.scatter(subset['PC1'], subset['PC2'], c=color, label=target, alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Iris Dataset Dimensionality Reduction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
