import numpy as np
from typing import Tuple

class BaseClassifier:
    """Base class for all classifiers in CCLASSY.
    Provides data validation and preparation for C API.
    Meant to be inherited by specific classifier classes.
    """
    
    def __init__(self):
        self.X = None
        self.y = None
        self.fitted = False
        
    def _validate_X_y(self, X: np.ndarray, y: np.ndarray) -> None:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be 1D array of shape (n_samples,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if not np.all((y==0) | (y==1)):
            raise ValueError("y must contain only binary lables: 0 or 1")
        
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int32)
        return X, y
    
    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (n_samples, n_features)")
        return np.ascontiguousarray(X, dtype=np.float64)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        raise NotImplementedError("fit method must be implemented by subclass")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("predict method must be implemented by subclass")