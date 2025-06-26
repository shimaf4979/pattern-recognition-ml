import numpy as np
from sklearn.datasets import make_regression



class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaN values")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Input data contains infinite values")

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            cost = np.mean((y - y_predicted) ** 2)
            self.cost_history.append(cost)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def plot_cost_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

    def plot_regression_line(self, X, y):
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', alpha=0.6)
        y_pred = self.predict(X)
        plt.plot(X, y_pred, color='red', linewidth=2)
        plt.title('Linear Regression Result')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.show()

if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    model = LinearRegression()
    model.fit(X, y)
    print(f"Final cost: {model.cost_history[-1]:.4f}")
