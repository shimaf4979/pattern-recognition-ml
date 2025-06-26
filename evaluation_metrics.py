import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class RegressionMetrics:
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def root_mean_squared_error(self, y_true, y_pred):
        return np.sqrt(self.mean_squared_error(y_true, y_pred))
    
    def mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

class ClassificationMetrics:
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def precision(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    def recall(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def f1_score(self, y_true, y_pred):
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
