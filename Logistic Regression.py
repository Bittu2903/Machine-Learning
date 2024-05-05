import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y, y_predicted):
        loss = -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        return loss
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for epoch in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            loss = self.compute_loss(y, y_predicted)
            self.losses.append(loss)
            # Removed print statement to avoid printing loss at each iteration
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = np.round(y_predicted).astype(int)
        return y_predicted_cls

    def plot_decision_boundary(self, X, y):
        x1 = np.linspace(np.min(X[:,0])-1, np.max(X[:,0])+1, 100)
        x2 = -(self.weights[0]*x1 + self.bias) / self.weights[1]

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolors='k', s=100)
        plt.plot(x1, x2, color='g', linewidth=2, label='Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.legend()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Example dataset
    X_train = np.array([[1, 2], [1, 5], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([0, 0, 1, 1, 1])

    X_test = np.array([[1, 1], [2, 2], [3, 1], [4, 4], [2.074, 5]])

    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = np.array(model.predict(X_test))
    print("Predicted Classes:", predictions)
    
    # Plotting the loss over iterations graph
    plt.plot(range(len(model.losses)), model.losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    plt.show()

    # Plotting the decision boundary
    model.plot_decision_boundary(X_test, predictions)
