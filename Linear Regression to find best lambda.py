import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Test data
X_test = np.array([1, 3, 5, 7, 9])
y_test = np.array([1, 3, 6, 7, 9])

# Training data
X = np.random.randint(1, 10, 100)
y = np.random.randint(1, 10, 100)

# Initial parameters
theta_0 = 10
theta_1 = 10

# Hyperparameter search
lambda_values = [0.001, 0.01, 0.1, 0.5, 1.0]  # Adjust the values based on your search space

best_lambda = None
min_test_error = float('inf')
test_errors = []

for lmda in lambda_values:
    N = len(X)
    Max_epoch = 100
    lr = 0.001 
    history = {}
    
    # Training loop
    print('\n')
    for epoch in range(Max_epoch):
        yhat = y - (theta_1 * X + theta_0)
        dtheta_1 = (1/N) * np.sum(X.dot(y - yhat)) + lmda * theta_1
        dtheta_0 = (1/N) * np.sum(y - yhat) + lmda * theta_0

        theta_1 = theta_1 - lr * dtheta_1 
        theta_0 = theta_0 - lr * dtheta_0 

        loss = (1/N) * np.sum(yhat)
        history[(theta_1, theta_0)] = loss
#         print(f"epoch {epoch+1} , theta_0 = {theta_0}, theta_1 = {theta_1}, loss = {loss} ")
    plt.scatter(X,y)
    plt.plot(X_test,(theta_1*X_test + theta_0), color='red')
    plt.show()
    
    # Evaluate on test data
    y_test_hat = theta_1 * X_test + theta_0
    test_error = np.mean((y_test - y_test_hat)**2)
    test_errors.append(test_error)
    # Record the best lambda
    if test_error < min_test_error:
        min_test_error = test_error
        best_lambda = lmda

    print(f"For lambda={lmda}, test error = {test_error}")

plt.scatter(test_errors,lambda_values)
plt.plot(test_errors,lambda_values, color='red')
plt.xlabel('test_errors')
plt.ylabel('lambda_values')
plt.show()
print(f"\nBest lambda: {best_lambda} with minimum test error: {min_test_error}")
