import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

w = 0.0
b = 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

learning_rate = 0.1
iterations = 1000

for i in range(iterations):
    z = w * X_train + b
    p = sigmoid(z)

    gradient_w = np.mean((p - Y_train) * X_train)
    gradient_b = np.mean(p - Y_train)

    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

z_test = w * X_test + b
p_test = sigmoid(z_test)

threshold = 0.5
predictions = (p_test >= threshold).astype(int)

print("Actual:", Y_test)
print("Predictions:", predictions)
print("Accuracy:", accuracy_score(Y_test, predictions) * 100, "%")