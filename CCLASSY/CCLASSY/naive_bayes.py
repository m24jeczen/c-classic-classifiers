import numpy as np
from .core import BaseClassifier
from .cmodule import py_nb_fit, py_nb_predict

class NaiveBayesClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.mean_ = None
        self.var_ = None
        self.prior_ = None
    
    def fit(self, X: np.ndarray, y:np.ndarray) -> 'NaiveBayesClassifier':
        self._validate_X_y(X, y)
        X, y = self._prepare_data(X, y)
        self.mean_, self.var_, self.prior_ = py_nb_fit(X,y)
        self.fitted = True
        return self
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit() before predict()")
        X = self._prepare_X(X)
        return py_nb_predict(self.mean_, self.var_, self.prior_, X)
