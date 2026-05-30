import numpy as np
from typing import Tuple
from .core import BaseClassifier
from .cmodule import py_knn_predict

class KNNClassifier(BaseClassifier):
    """
blabla
    """
    
    def __init__(self, k: int = 3):
        super().__init__()
        if k <= 0:
            raise ValueError("k must be >= 0")
        self.k = k
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        self._validate_X_y(X, y)
        if self.k > X.shape[0]:
            raise ValueError("k must be less than or equal to the number of training samples")
        self.X, self.y = self._prepare_data(X,y)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("call fit() before predict()")
        X = self._prepare_X(X)
        if X.shape[1] != self.X.shape[1]:
            raise ValueError(f"X has {X.shape[1]} features, but expected {self.X.shape[1]}")
        return py_knn_predict(self.X, self.y, X, self.k)
    
    
    
    