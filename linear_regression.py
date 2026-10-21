import numpy as np



X = np.array([1,2,3,4,5,6,7,8])
Y = np.array([52,55,63,70,78,85,88,95])

w = 5.2175
b = 50

iterations = 100
for i in range(iterations):
    prediction = w * X + b
    error = prediction - Y 
    mse = np.mean(error ** 2)
    
    gradient_w = (2 / len(X)) * np.sum(error * X)
    gradient_b = (2 / len(X)) * np.sum(error)

    w = w - 0.01 * gradient_w
    b = b - 0.01 * gradient_b


print("w:", w)
print("b:", b)
print("MSE:", mse)



