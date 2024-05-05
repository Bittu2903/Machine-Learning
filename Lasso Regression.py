import numpy as np
import matplotlib.pyplot as plt

def linear_regression_lasso_gradient_descent_regularized(X, y, learning_rate, epochs, lambda_reg):
    m, n = X.shape
    theta = np.zeros((n, 1))

    for epoch in range(epochs):
        # Hypothesis
        h = np.dot(X, theta)

        # Calculate error
        error = h - y

        # Calculate gradients with L1 regularization (Lasso)
        sign = np.sign(theta)
        sign[0] = 0  # No regularization for bias term
        gradient = (np.dot(X.T, error) + lambda_reg * sign) / m

        # Update parameters
        theta -= learning_rate * gradient

    # Calculate final cost (mean squared error with L1 regularization)
    final_cost = np.sum((h - y) ** 2) / (2 * m) + (lambda_reg * np.sum(np.abs(theta[1:]))) / m

    return theta, final_cost


def find_best_lambda(X, y, learning_rate, epochs, lambda_values):
    costs = []

    for lambda_reg in lambda_values:
        # Add bias term to X
        X_b = np.c_[np.ones((m, 1)), X]

        # Perform regularized linear regression with gradient descent
        theta, final_cost = linear_regression_lasso_gradient_descent_regularized(X_b, y, learning_rate, epochs, lambda_reg)

        # Store the final cost for the current lambda
        costs.append(final_cost)

    # Find the lambda with the minimum cost
    best_lambda = lambda_values[np.argmin(costs)]

    return best_lambda, costs

# Example usage:
# Assuming X is your feature matrix and y is your target variable
X = np.random.rand(100, 1)  # Example data, replace with your actual data
y = 4 + 3 * X + 0.1 * np.random.randn(100, 1)  # Example target, replace with your actual data

# Define the number of training examples
m = X.shape[0]

# Set hyperparameters
learning_rate = 0.01
epochs = 1000

# Define a range of lambda values to test
lambda_values = np.logspace(-5, 2, 100)  # Adjust the range as needed

# Find the best lambda and corresponding costs
best_lambda, costs = find_best_lambda(X, y, learning_rate, epochs, lambda_values)

# Plot the costs for different lambda values
plt.plot(lambda_values, costs, label='Cost')
plt.scatter(best_lambda, min(costs), color='red', label=f'Best Lambda: {best_lambda}', marker='x')
plt.xscale('log')  # Log scale for better visualization
plt.xlabel('Lambda (Regularization Parameter)')
plt.ylabel('Cost')
plt.title('Cost vs. Lambda')
plt.legend()
plt.show()

print(f'Best Lambda: {best_lambda}')
