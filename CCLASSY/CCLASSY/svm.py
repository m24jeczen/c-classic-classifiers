import numpy as np
from .core import BaseClassifier
from .cmodule import py_svm_fit, py_svm_predict, py_svm_free

class SVMClassifier(BaseClassifier):
    def __init__(self, n_iterations: int=1000,
                 learning_rate: float=0.01,
                 lambda_: float=0.01):
        super().__init__()
        self._model = None
        if n_iterations <= 0:
            raise ValueError("n iterations must be > 0")
        if learning_rate <= 0.0:
            raise ValueError("learning rate must be > 0")
        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0")
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        
    def fit(self, X:np.ndarray, y:np.ndarray) -> 'SVMClassifier':
        self._validate_X_y(X,y)
        X,y = self._prepare_data(X,y)
        if self._model is not None:
            py_svm_free(self._model)
        self._model = py_svm_fit(X,y, self.n_iterations, self.learning_rate, self.lambda_)
        self.fitted = True
        return self
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit() before predict()")
        X = self._prepare_X(X)
        return py_svm_predict(self._model, X)
    
    def __del__(self):
        if self._model is not None:
            py_svm_free(self._model)
            self._model = None