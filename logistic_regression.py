import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        z = np.clip(z, -250, 250)  # オーバーフロー防止
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            cost = self.compute_cost(y, y_predicted)
            self.cost_history.append(cost)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def compute_cost(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        predictions = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(predictions)

if __name__ == "__main__":
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=42)
    model = LogisticRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy:.4f}")
