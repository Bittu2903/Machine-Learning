import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_points = 100
np.random.seed(0)
X = np.random.normal(0, 1, num_points)
Y = 2 * X + 3 + np.random.normal(0, 0.5, num_points)

def linear_regression(X, Y, learning_rate):
    # slope & intercept
    m = np.random.normal()
    c = np.random.normal()
    print(f'slope = {m}')
    print(f'intercept = {c}')
    # loss value
    loss = m
    itr = 0
    while(True):
        y_pred = (m * X) + c
        gradient_m = (-2/num_points) * np.dot(X, (Y - y_pred))
        gradient_c = (-2/num_points) * np.sum(Y - y_pred)
        # print(f'{itr}-gradient_m: {gradient_m}')
        m = m - learning_rate * gradient_m
        c = c - learning_rate * gradient_c
        if(m == loss):
            break
        loss = m
        itr += 1
    return m, c

learning_rate = 0.01
slope, intercept = linear_regression(X, Y, learning_rate)

plt.scatter(X, Y, label = "data_points")
plt.plot(X, slope * X + intercept, color = "red", label = "regression_line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()