import time
import numpy as np
from sklearn.datasets import make_regression, make_classification
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression

class ModelComparison:
    def __init__(self):
        self.results = {}
    
    def compare_models(self, X_reg, y_reg, X_clf, y_clf):
        # 線形回帰の性能測定
        start_time = time.time()
        lr_model = LinearRegression()
        lr_model.fit(X_reg, y_reg)
        lr_time = time.time() - start_time
        lr_pred = lr_model.predict(X_reg)
        lr_mse = np.mean((y_reg - lr_pred) ** 2)
        
        # ロジスティック回帰の性能測定  
        start_time = time.time()
        lg_model = LogisticRegression()
        lg_model.fit(X_clf, y_clf)
        lg_time = time.time() - start_time
        lg_pred = lg_model.predict(X_clf)
        lg_accuracy = np.mean(lg_pred == y_clf)
        
        self.results = {
            'linear_regression': {'time': lr_time, 'mse': lr_mse},
            'logistic_regression': {'time': lg_time, 'accuracy': lg_accuracy}
        }
        return self.results

if __name__ == "__main__":
    X_reg, y_reg = make_regression(n_samples=200, n_features=1, random_state=42)
    X_clf, y_clf = make_classification(n_samples=200, n_features=2, random_state=42)
    
    comparison = ModelComparison()
    results = comparison.compare_models(X_reg, y_reg, X_clf, y_clf)
    print("Performance Comparison:", results)
