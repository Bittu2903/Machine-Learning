import numpy as np
import matplotlib.pyplot as plt

# Function to calculate mean vectors for each class
def calculate_mean_vectors(X, y):
    classes = np.unique(y)
    means = []
    for c in classes:
        means.append(np.mean(X[y == c], axis=0))
    return means

# Function to compute the within-class scatter matrix
def within_class_scatter_matrix(X, y):
    classes = np.unique(y)
    n_features = X.shape[1]
    SW = np.zeros((n_features, n_features))  # Initialize within-class scatter matrix
    means = calculate_mean_vectors(X, y)  # Calculate mean vectors for each class
    for c, mean in zip(classes, means):
        class_scatter = np.zeros((n_features, n_features))
        # Compute scatter matrix for each class
        for row in X[y == c]:
            row, mean = row.reshape(n_features, 1), mean.reshape(n_features, 1)
            class_scatter += (row - mean).dot((row - mean).T)
        # Accumulate class scatter matrices
        SW += class_scatter
    # Add regularization term for numerical stability
    SW += np.eye(n_features) * 1e-8
    return SW

# Function to compute the between-class scatter matrix
def between_class_scatter_matrix(X, y):
    classes = np.unique(y)
    n_features = X.shape[1]
    SB = np.zeros((n_features, n_features))  # Initialize between-class scatter matrix
    means = calculate_mean_vectors(X, y)  # Calculate mean vectors for each class
    overall_mean = np.mean(X, axis=0).reshape(n_features, 1)  # Calculate overall mean
    for c, mean in zip(classes, means):
        n = X[y == c].shape[0]  # Number of samples in class
        mean = mean.reshape(n_features, 1)
        overall_mean = overall_mean.reshape(n_features, 1)
        # Compute scatter matrix for each class
        SB += n * (mean - overall_mean).dot((mean - overall_mean).T)
    return SB

# Function to perform Linear Discriminant Analysis (LDA)
def lda(X, y, n_components=1):
    SW = within_class_scatter_matrix(X, y)  # Compute within-class scatter matrix
    SB = between_class_scatter_matrix(X, y)  # Compute between-class scatter matrix
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(SW).dot(SB))  # Solve generalized eigenvalue problem
    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvectors[:, :n_components]

# Function to plot the original data
def plot_data(X, y, title):
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        plt.scatter(X[y == label, 0], X[y == label, 1], label=label)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Function to plot the LDA projection
def plot_lda_projection(X_lda, y):
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        plt.scatter(X_lda[y == label], np.zeros_like(X_lda[y == label]), label=label)
    plt.title('LDA Projection')
    plt.xlabel('LD1')
    plt.legend()
    plt.show()

# Example usage
X = np.array([[4, 2], [2, 4], [2, 3], [3, 3], [4, 4]])
y = np.array([0, 0, 1, 1, 1])

# Plot data before LDA
plot_data(X, y, title='Data Before LDA')

# Perform LDA and plot the projection
X_lda = X.dot(lda(X, y))
plot_lda_projection(X_lda, y)
