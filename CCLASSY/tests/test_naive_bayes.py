import pytest
import numpy as np
from CCLASSY import NaiveBayesClassifier

@pytest.fixture
def simple_data():
    X_train = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [1.2, 2.1],
        [5.0, 6.0],
        [5.5, 5.8],
        [5.2, 6.1]
    ])
    y_train = np.array([0,0,0,1,1,1])
    return X_train, y_train


def test_fit_returns_self(simple_data):
    X,y = simple_data
    clf = NaiveBayesClassifier()
    assert clf.fit(X,y) is clf
    
def test_fit_sets_fitted(simple_data):
    X,y = simple_data
    clf = NaiveBayesClassifier()
    assert clf.fitted == False
    clf.fit(X,y)
    assert clf.fitted == True
    
def test_fit_invalid_X_not_2d(simple_data):
    _, y = simple_data
    with pytest.raises(ValueError):
        NaiveBayesClassifier().fit(np.array([1,2,3]), y)
        
def test_fit_invalid_y_not_binary(simple_data):
    X, _ = simple_data
    with pytest.raises(ValueError):
        NaiveBayesClassifier().fit(X, np.array([0,1,2,0,1,2]))
        
        
        
def test_predict_before_fit():
    with pytest.raises(RuntimeError):
        NaiveBayesClassifier().predict(np.array([[1.0, 2.0]]))
      
def test_predict_correct_classes(simple_data):
    X_train, y_train = simple_data
    X_test = np.array([[1.1, 2.0], [5.1, 6.0]])
    clf = NaiveBayesClassifier().fit(X_train, y_train)
    np.testing.assert_array_equal(clf.predict(X_test), np.array([0,1]))
    
def test_predict_output_shape(simple_data):
    X_train, y_train = simple_data
    X_test = np.array([[1.1, 2.0], [5.1, 6.0], [1.3, 1.9]])
    clf = NaiveBayesClassifier().fit(X_train, y_train)
    assert clf.predict(X_test).shape == (3,)
    
def test_predict_output_binary(simple_data):
    X_train, y_train = simple_data
    X_test = np.array([[1.1, 2.0], [5.1, 6.0]])
    clf = NaiveBayesClassifier().fit(X_train, y_train)
    assert set(clf.predict(X_test)).issubset({0,1})
    
def test_predict_wrong_n_features(simple_data):
    X_train, y_train = simple_data
    clf = NaiveBayesClassifier().fit(X_train, y_train)
    with pytest.raises(ValueError):
        clf.predict(np.array([[1.0, 2.0, 3.0]]))
        
def test_refit(simple_data):
    X,y = simple_data
    clf = NaiveBayesClassifier().fit(X,y).fit(X,y)
    assert clf.fitted == True
    
def test_method_chaining(simple_data):
    X_train, y_train = simple_data
    X_test = np.array([[1.1, 2.0]])
    assert NaiveBayesClassifier().fit(X_train, y_train).predict(X_test)[0] == 0
        
        
