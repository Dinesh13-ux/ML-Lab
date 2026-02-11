
import numpy as np
import matplotlib.pyplot as plt

# Example dataset (Hours studied vs Exam score)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

n = len(X)

w = 0
b = 0

learning_rate = 0.01
iterations = 1000

for i in range(iterations):
    
    # Predictions
    y_pred = w * X + b
    
    # Compute gradients
    dw = (2/n) * np.sum((y_pred - y) * X)
    db = (2/n) * np.sum(y_pred - y)
    
    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

print("Final weight:", w)
print("Final bias:", b)

# Predict new value
x_new = 6
prediction = w * x_new + b
print("Prediction for x=6:", prediction)


plt.scatter(X, y, color='blue', label="Actual Data")

plt.plot(X, w*X + b, color='red', label="Regression Line")

plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
