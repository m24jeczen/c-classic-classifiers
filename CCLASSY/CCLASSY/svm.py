import numpy as np
from .core import BaseClassifier
from .cmodule import py_svm_fit, py_svm_predict

class SVMClassifier(BaseClassifier):
    def __init__(self, n_iterations: int=1000,
                 learning_rate: float=0.01,
                 lambda_: float=0.01):
        super().__init__()
        if n_iterations <= 0:
            raise ValueError("n iterations must be > 0")
        if learning_rate <= 0.0:
            raise ValueError("learning rate must be > 0")
        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0")
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X:np.ndarray, y:np.ndarray) -> 'SVMClassifier':
        self._validate_X_y(X,y)
        X,y = self._prepare_data(X,y)
        self.coef_, self.intercept_ = py_svm_fit(X,y, self.n_iterations, self.learning_rate, self.lambda_)
        self.fitted = True
        return self
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit() before predict()")
        X = self._prepare_X(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.coef_.shape[0]}"
            )
        return py_svm_predict(self.coef_, self.intercept_, X)
