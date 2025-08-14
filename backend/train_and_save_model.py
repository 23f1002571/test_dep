import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Example training data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

# Save trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model.pkl")
