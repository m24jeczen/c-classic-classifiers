import pytest
import numpy as np
from CCLASSY import KNNClassifier

@pytest.fixture
def simple_data():
    """Two visibly separated clusters - class 0 close to (1,2), class 1 close to (5,6)
    """
    X_train = np.array([
        [1.0, 2.0], 
        [1.5, 1.8],
        [1.2, 2.1],
        [5.0, 6.0],
        [5.5, 5.8],
        [5.2, 6.1]
    ])
    
    y_train = np.array([0, 0,0,1,1,1])
    return X_train, y_train

def test_invalid_k_zero():
    with pytest.raises(ValueError):
        KNNClassifier(k=0)
        
def test_invalid_k_negative():
    with pytest.raises(ValueError):
        KNNClassifier(k=-1)
        
def test_default_k():
    clf = KNNClassifier()
    assert clf.k == 3
    

def test_fit_returns_self(simple_data):
    X,y = simple_data
    clf = KNNClassifier(k=3)
    result = clf.fit(X,y)
    assert result is clf

def test_fit_sets_fitted(simple_data):
    X,y = simple_data
    clf = KNNClassifier(k=3)
    assert clf.fitted == False
    clf.fit(X,y)
    assert clf.fitted == True
    
def test_fit_invalid_X_not_2d(simple_data):
    _, y = simple_data
    with pytest.raises(ValueError):
        KNNClassifier(k=3).fit(np.array([1.0, 2.0, 3.0]), y)

def test_fit_invalid_y_not_1d(simple_data):
    X, _ = simple_data
    with pytest.raises(ValueError):
        KNNClassifier(k=3).fit(X, np.array([[0,1], [0,1]]))
        
def test_fit_invalid_y_not_binary(simple_data):
    X, _ = simple_data
    y_bad = np.array([0,1,2,0,1,2])
    with pytest.raises(ValueError):
        KNNClassifier(k=3).fit(X, y_bad)
        
def test_fit_mismatched_samples(simple_data):
    X, _ = simple_data 
    y_bad = np.array([0,1,0])
    with pytest.raises(ValueError):
        KNNClassifier(k=3).fit(X, y_bad)
        
def test_fit_k_larger_than_n_samples(simple_data):
    X,y = simple_data
    with pytest.raises(ValueError):
        KNNClassifier(k=100).fit(X,y)
        
    
def test_predict_before_fit():
    clf = KNNClassifier(k=3)
    X_test = np.array([[1.0, 2.0]])
    with pytest.raises(RuntimeError):
        clf.predict(X_test)
        
def test_predict_output_shape(simple_data):
    X_train, y_train = simple_data
    X_test = np.array([[1.1, 2.0], [5.1, 6.0], [1.3, 1.9]])
    clf = KNNClassifier(k=3).fit(X_train, y_train)
    predictions = clf.predict(X_test)
    assert predictions.shape == (3,)
    
def test_predict_output_binary(simple_data):
    X_train, y_train = simple_data
    X_test = np.array([[1.1, 2.0], [5.1, 6.0]])
    clf = KNNClassifier(k=3).fit(X_train, y_train)
    predictions = clf.predict(X_test)
    assert set(predictions).issubset({0,1})
    
def test_predict_wrong_n_features(simple_data):
    X_train, y_train = simple_data
    clf = KNNClassifier(k=3).fit(X_train, y_train)
    X_test_bad = np.array([[1.1, 2.0, 3.0]])
    with pytest.raises(ValueError):
        clf.predict(X_test_bad)
    
def test_predict_k1_nearest_neighbor(simple_data):
    X_train, y_train = simple_data
    clf = KNNClassifier(k=1). fit(X_train, y_train)
    X_test = np.array([[1.0, 2.0]])
    predictions = clf.predict(X_test)
    assert predictions[0] ==0
    
def test_method_chaining(simple_data):
    X_train, y_train = simple_data
    X_test = np.array([[1.1, 2.0]])
    predictions = KNNClassifier(k=3).fit(X_train, y_train).predict(X_test)
    assert predictions[0] == 0