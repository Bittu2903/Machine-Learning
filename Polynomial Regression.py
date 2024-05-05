import numpy as np
import matplotlib.pyplot as plt

def generate_sample_data(seed=42):
    print('----->>>>',degree)
    np.random.seed(seed)
    X = 2 * np.random.rand(50, 1)
    y = 1.5 * X**degree + np.random.randn(50, 1)
    return X, y

def polynomial_regression(X, y, degree, learning_rate, num_epochs):
    # Initialize parameters
    theta = np.random.randn(degree + 1, 1)  # Adjusted to match the degree
    gradient_history = []
    MSEs = []
    Epochs = []

    for epoch in range(num_epochs):
        X_poly = add_polynomial_features(X, degree)
        y_pred = X_poly.dot(theta)

        # Compute mean squared error
        mse = calculate_mse(y_pred, y)
        
        # Compute gradients
        gradients = compute_gradients(X_poly, y_pred, y)

        # Update parameters using gradient descent
        theta = update_parameters(theta, learning_rate, gradients)

        # Store values for visualization
        Epochs.append(epoch)
        MSEs.append(mse)
        gradient_history.append(np.linalg.norm(gradients))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE: {mse}")

    # Plot the change in gradients over epochs
    plot_gradient_changes(Epochs, gradient_history)

    return theta, Epochs, MSEs

def add_polynomial_features(X, degree):
    X_poly = np.c_[X]
    for d in range(2, degree + 1):
        X_poly = np.c_[X_poly, X**d]
    return np.c_[np.ones((X.shape[0], 1)), X_poly]

def calculate_mse(y_pred, y):
    return np.mean((y_pred - y)**2)

def compute_gradients(X_poly, y_pred, y):
    return 2/X_poly.shape[0] * X_poly.T.dot(y_pred - y)

def update_parameters(theta, learning_rate, gradients):
    return theta - learning_rate * gradients

def plot_gradient_changes(Epochs, gradient_history):
    plt.plot(Epochs, gradient_history, label='Change in Gradients')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Magnitude')
    plt.title('Change in Gradients Over Epochs')
    plt.legend()
    plt.show()

def generate_and_plot_predictions(X, y, degree, learning_rate, num_epochs):
    theta, _, _ = polynomial_regression(X, y, degree, learning_rate, num_epochs)

    X_new = np.linspace(0, 2, 50).reshape(50, 1)
    X_new_poly = add_polynomial_features(X_new, degree)
    y_pred = X_new_poly.dot(theta)

    plt.scatter(X, y, label='Original Data')
    plt.plot(X_new, y_pred, 'r-', label=f'Polynomial Regression (Degree {degree})', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def plot_mse_over_epochs(X, y, degree, learning_rate, num_epochs):
    _, Epochs, MSEs = polynomial_regression(X, y, degree, learning_rate, num_epochs)

    plt.plot(Epochs, MSEs, label='MSEs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

# Usage
degree = 3
X, y = generate_sample_data(degree)
learning_rate = 0.01
num_epochs = 1000

generate_and_plot_predictions(X, y, degree, learning_rate, num_epochs)

plot_mse_over_epochs(X, y, degree, learning_rate, num_epochs)