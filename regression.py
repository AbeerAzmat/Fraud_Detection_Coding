# Iris Regression Script (Synthetic Cubic Data)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic non-linear data
x = np.random.rand(40, 1)
y = x ** 3 + np.random.rand(40, 1) / 5

# Train linear regression model
model = LinearRegression()
model.fit(x, y)

# Predict on smooth input range
xx = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = model.predict(xx)

# Plot original data and linear regression
plt.figure(figsize=(6, 4))
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(xx, y_pred, '--r', label='Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression: Linear Fit on Cubic Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Mean Squared Error
mse = mean_squared_error(y, model.predict(x))
print("Mean Squared Error:", mse)
