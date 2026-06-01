import numpy as np
from .core import BaseClassifier
from .cmodule import py_nb_fit, py_nb_predict, py_nb_free

class NaiveBayesClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self._model = None
    
    def fit(self, X: np.ndarray, y:np.ndarray) -> 'NaiveBayesClassifier':
        self._validate_X_y(X, y)
        X, y = self._prepare_data(X, y)
        if self._model is not None:
            py_nb_free(self._model)
        self._model = py_nb_fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit() before predict()")
        X = self._prepare_X(X)
        return py_nb_predict(self._model, X)
    
    def __del__(self):
        if self._model is not None:
            py_nb_free(self._model)
            self._model = None