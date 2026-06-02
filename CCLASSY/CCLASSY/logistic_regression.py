import numpy as np
from .core import BaseClassifier
from .cmodule import py_lr_fit, py_lr_predict, py_lr_free

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, n_iterations: int=1000, learning_rate: float=0.1):
        super().__init__()
        if n_iterations <= 0:
            raise ValueError("n iterations must be > 0")
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._model = None
        
    def fit(self, X:np.ndarray, y:np.ndarray) -> 'LogisticRegressionClassifier':
        self._validate_X_y(X,y)
        X,y = self._prepare_data(X,y)
        if self._model is not None:
            py_lr_free(self._model)
        self._model = py_lr_fit(X,y, self.n_iterations, self.learning_rate)
        self.fitted = True
        return self
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit() before predict()")
        X = self._prepare_X(X)
        return py_lr_predict(self._model, X)
    
    def __del__(self):
        if self._model is not None:
            py_lr_free(self._model)
            self._model = None
            
            